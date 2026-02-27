#include "cookie_ekf.h"
#include <string.h>
#include <math.h>

#define PI 3.141592653589793f
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

// WGS84 Constants
static const float a = 6378137.0f;
// static const float f = 1.0f / 298.257223563f;
static const float b = 6356752.3142f; // a * (1 - f)
static const float e2 = 0.00669437999f; // eccentricity^2 (1-b^2/a^2)
static const float ep2 = 0.00673949674f; // second eccentricity^2 (a^2/b^2 -1)

// Gauss-Markov Coefficients
static const float beta_acc = 3.7e-7f;
static const float beta_gyr = 2.9e-1f;

// Helper: Initialize Identity Matrix
static void mat_eye(arm_matrix_instance_f32 *mat, float val) {
    uint16_t size = mat->numRows * mat->numCols;
    memset(mat->pData, 0, size * sizeof(float));
    for (int i = 0; i < mat->numRows; i++) {
        mat->pData[i * mat->numCols + i] = val;
    }
}

// Helper: Set matrix to zero
static void mat_zero(arm_matrix_instance_f32 *mat) {
    uint16_t size = mat->numRows * mat->numCols;
    memset(mat->pData, 0, size * sizeof(float));
}

// May be need double if precision is not enough
void LLA_to_ENU(float lat, float lon, float alt, float lat0, float lon0, float alt0, float *enu) {
    // 1. LLA to ECEF
    float phi = lat * DEG2RAD;
    float lam = lon * DEG2RAD;
    float sphi = sinf(phi);
    float cphi = cosf(phi);
    float slam = sinf(lam);
    float clam = cosf(lam);
    
    float N = a / sqrtf(1.0f - e2 * sphi * sphi);
    float x = (N + alt) * cphi * clam;
    float y = (N + alt) * cphi * slam;
    float z = (N * (1.0f - e2) + alt) * sphi;

    // Reference ECEF
    float phi0 = lat0 * DEG2RAD;
    float lam0 = lon0 * DEG2RAD;
    float sphi0 = sinf(phi0);
    float cphi0 = cosf(phi0);
    float slam0 = sinf(lam0);
    float clam0 = cosf(lam0);
    
    float N0 = a / sqrtf(1.0f - e2 * sphi0 * sphi0);
    float x0 = (N0 + alt0) * cphi0 * clam0;
    float y0 = (N0 + alt0) * cphi0 * slam0;
    float z0 = (N0 * (1.0f - e2) + alt0) * sphi0;

    // 2. ECEF Delta
    float dx = x - x0;
    float dy = y - y0;
    float dz = z - z0;

    // 3. ECEF to ENU Rotation
    // R = [ -sin(lam0)           cos(lam0)          0
    //       -sin(phi0)cos(lam0)  -sin(phi0)sin(lam0) cos(phi0)
    //        cos(phi0)cos(lam0)   cos(phi0)sin(lam0) sin(phi0) ]
    
    enu[0] = -slam0 * dx + clam0 * dy;
    enu[1] = -sphi0 * clam0 * dx - sphi0 * slam0 * dy + cphi0 * dz;
    enu[2] = cphi0 * clam0 * dx + cphi0 * slam0 * dy + sphi0 * dz;
}

// May be need double if precision is not enough
void ENU_to_LLA(float *enu, float lat0, float lon0, float alt0, float *lla) {
    // 1. ENU to ECEF
    float phi0 = lat0 * DEG2RAD;
    float lam0 = lon0 * DEG2RAD;
    float sphi0 = sinf(phi0);
    float cphi0 = cosf(phi0);
    float slam0 = sinf(lam0);
    float clam0 = cosf(lam0);

    float N0 = a / sqrtf(1.0f - e2 * sphi0 * sphi0);
    float x0 = (N0 + alt0) * cphi0 * clam0;
    float y0 = (N0 + alt0) * cphi0 * slam0;
    float z0 = (N0 * (1.0f - e2) + alt0) * sphi0;

    float dx = -slam0 * enu[0] - sphi0 * clam0 * enu[1] + cphi0 * clam0 * enu[2];
    float dy =  clam0 * enu[0] - sphi0 * slam0 * enu[1] + cphi0 * slam0 * enu[2];
    float dz =                   cphi0 * enu[1] + sphi0 * enu[2];

    float x = x0 + dx;
    float y = y0 + dy;
    float z = z0 + dz;

    // 2. ECEF to LLA (Bowring's method)
    float p = sqrtf(x*x + y*y);
    float theta = atan2f(z * a, p * b);
    float sintheta = sinf(theta);
    float costheta = cosf(theta);
    float sintheta3 = sintheta * sintheta * sintheta;
    float costheta3 = costheta * costheta * costheta;

    float phi = atan2f(z + ep2 * b * sintheta3, p - e2 * a * costheta3);
    float lam = atan2f(y, x);

    float sphi = sinf(phi);
    float N = a / sqrtf(1.0f - e2 * sphi * sphi);
    
    lla[0] = phi * RAD2DEG;
    lla[1] = lam * RAD2DEG;
    lla[2] = p / cosf(phi) - N;
}

void EKF_Init(EKF_Context_t *ekf, float lat0, float lon0, float alt0) {
    memset(ekf, 0, sizeof(EKF_Context_t));
    
    ekf->lla0[0] = lat0;
    ekf->lla0[1] = lon0;
    ekf->lla0[2] = alt0;

    // Initialize Matrix Instances
    arm_mat_init_f32(&ekf->P, 15, 15, ekf->P_data);
    arm_mat_init_f32(&ekf->Q, 15, 15, ekf->Q_data);
    arm_mat_init_f32(&ekf->R, 3, 3, ekf->R_data);
    arm_mat_init_f32(&ekf->x, 15, 1, ekf->x_data);

    // Initialize Q (Process Noise)
    float Qpos = 2.0f;
    float Qvel = 2.0f;
    float Qorient[3] = {0.0002f, 0.0002f, 0.2f};
    float Qacc = 0.1f;
    float Qgyr[3] = {0.0001f, 0.0001f, 0.1f};

    mat_zero(&ekf->Q);
    for(int i=0; i<3; i++) ekf->Q_data[i*15 + i] = Qpos;
    for(int i=3; i<6; i++) ekf->Q_data[i*15 + i] = Qvel;
    for(int i=6; i<9; i++) ekf->Q_data[i*15 + i] = Qorient[i-6];
    for(int i=9; i<12; i++) ekf->Q_data[i*15 + i] = Qacc;
    for(int i=12; i<15; i++) ekf->Q_data[i*15 + i] = Qgyr[i-12];

    // Initialize R (Measurement Noise)
    float Rpos = 2.0f;
    mat_eye(&ekf->R, Rpos);

    // Initialize P (Covariance)
    mat_eye(&ekf->P, 0.0f); // Start with 0 as per Python script

    // Initial Biases
    ekf->acc_bias[0] = ekf->acc_bias[1] = ekf->acc_bias[2] = 1e-6f;
    ekf->gyr_bias[0] = ekf->gyr_bias[1] = ekf->gyr_bias[2] = 1e-7f;
    
    // Set initial state bias
    ekf->x_data[9] = ekf->acc_bias[0];
    ekf->x_data[10] = ekf->acc_bias[1];
    ekf->x_data[11] = ekf->acc_bias[2];
    ekf->x_data[12] = ekf->gyr_bias[0];
    ekf->x_data[13] = ekf->gyr_bias[1];
    ekf->x_data[14] = ekf->gyr_bias[2];
}

void EKF_Predict(EKF_Context_t *ekf, float *acc_raw, float *gyr_raw, float dt) {
    // 1. Correct IMU data
    float acc[3], gyr[3];
    for(int i=0; i<3; i++) {
        acc[i] = acc_raw[i] + ekf->x_data[9+i];
        gyr[i] = gyr_raw[i] + ekf->x_data[12+i];
    }

    // 2. Navigation Equations
    float yaw = ekf->orient_ypr[0];
    float pitch = ekf->orient_ypr[1];
    float roll = ekf->orient_ypr[2];
    
    float cy = cosf(yaw); float sy = sinf(yaw);
    float cp = cosf(pitch); float sp = sinf(pitch);
    float cr = cosf(roll); float sr = sinf(roll);
    float tp = tanf(pitch);

    // Rbn (Body to Nav)
    float Rbn[9] = {
        cy, -sy, 0,
        sy,  cy, 0,
         0,   0, 1
    };

    // Acc in ENU
    float accENU[3];
    accENU[0] = Rbn[0]*acc[0] + Rbn[1]*acc[1] + Rbn[2]*acc[2];
    accENU[1] = Rbn[3]*acc[0] + Rbn[4]*acc[1] + Rbn[5]*acc[2];
    accENU[2] = Rbn[6]*acc[0] + Rbn[7]*acc[1] + Rbn[8]*acc[2];

    // Update Position & Velocity
    float g_vec[3] = {0, 0, 9.81f};
    for(int i=0; i<3; i++) {
        ekf->pos_enu[i] += dt * ekf->vel_enu[i] + 0.5f * dt * dt * (accENU[i] - g_vec[i]);
        ekf->vel_enu[i] += dt * (accENU[i] - g_vec[i]);
    }

    // Update Orientation
    float omega_yaw   = (sr/cp)*gyr[1] + (cr/cp)*gyr[2]; // Python matrix row 0 is [0, sin(r)/cos(p), cos(r)/cos(p)]
    float omega_pitch = cr*gyr[1] - sr*gyr[2];           // Python matrix row 1 is [0, cos(r), -sin(r)]
    float omega_roll  = gyr[0] + (sr*tp)*gyr[1] + (cr*tp)*gyr[2]; // Python matrix row 2 is [1, sin(r)tan(p), cos(r)tan(p)]

    ekf->orient_ypr[0] += dt * omega_yaw;
    ekf->orient_ypr[1] += dt * omega_pitch;
    ekf->orient_ypr[2] += dt * omega_roll;

    // Normalize Yaw (-PI to PI)
    if(ekf->orient_ypr[0] < -PI) ekf->orient_ypr[0] += 2*PI;
    if(ekf->orient_ypr[0] >= PI) ekf->orient_ypr[0] -= 2*PI;

    // 3. Jacobian F Construction
    float F_data[15*15];
    arm_matrix_instance_f32 F;
    arm_mat_init_f32(&F, 15, 15, F_data);
    mat_zero(&F);

    // Calculate M and N
    float llaIMU[3];
    ENU_to_LLA(ekf->pos_enu, ekf->lla0[0], ekf->lla0[1], ekf->lla0[2], llaIMU);
    float lat_rad = llaIMU[0] * DEG2RAD;
    float alt = llaIMU[2];
    float slat = sinf(lat_rad);
    float slat2 = slat * slat;
    
    float M = a * (1.0f - e2) / powf(1.0f - e2 * slat2, 1.5f);
    float N = a / sqrtf(1.0f - e2 * slat2);

    float accE = accENU[0];
    float accN = accENU[1];
    float accU = accENU[2];

    // F[0:3, 3:6] = I
    F_data[0*15 + 3] = 1.0f; F_data[1*15 + 4] = 1.0f; F_data[2*15 + 5] = 1.0f;
    
    // F[3,7] = accU; F[3,8] = -accN;
    F_data[3*15 + 7] = accU; F_data[3*15 + 8] = -accN;
    
    // F[4,6] = -accU; F[4,8] = accE;
    F_data[4*15 + 6] = -accU; F_data[4*15 + 8] = accE;
    
    // F[5,6] = accN; F[5,7] = -accE;
    F_data[5*15 + 6] = accN; F_data[5*15 + 7] = -accE;
    
    // F[6,4] = 1/(M+alt)
    F_data[6*15 + 4] = 1.0f / (M + alt);
    
    // F[7,3] = 1/(N+alt)
    F_data[7*15 + 3] = 1.0f / (N + alt);
    
    // F[8,3] = tan(lat)/(N+alt)
    F_data[8*15 + 3] = tanf(lat_rad) / (N + alt);
    
    // F[3:6, 9:12] = Rbn
    for(int r=0; r<3; r++) {
        for(int c=0; c<3; c++) {
            F_data[(3+r)*15 + (9+c)] = Rbn[r*3 + c];
        }
    }
    
    // F[6:9, 12:15] = Rbn
    for(int r=0; r<3; r++) {
        for(int c=0; c<3; c++) {
            F_data[(6+r)*15 + (12+c)] = Rbn[r*3 + c];
        }
    }
    
    // F[9:12, 9:12] = -beta_acc * I
    F_data[9*15 + 9] = -beta_acc; F_data[10*15 + 10] = -beta_acc; F_data[11*15 + 11] = -beta_acc;
    
    // F[12:15, 12:15] = -beta_gyr * I
    F_data[12*15 + 12] = -beta_gyr; F_data[13*15 + 13] = -beta_gyr; F_data[14*15 + 14] = -beta_gyr;

    // Discretize F: F = I + F * dt
    for(int i=0; i<15*15; i++) {
        F_data[i] *= dt;
    }
    for(int i=0; i<15; i++) {
        F_data[i*15 + i] += 1.0f;
    }

    // 4. Predict Covariance: P = F * P * F' + Q
    float tmp1_data[15*15];
    arm_matrix_instance_f32 tmp1;
    arm_mat_init_f32(&tmp1, 15, 15, tmp1_data);
    
    float Ft_data[15*15];
    arm_matrix_instance_f32 Ft;
    arm_mat_init_f32(&Ft, 15, 15, Ft_data);
    
    // tmp1 = F * P
    arm_mat_mult_f32(&F, &ekf->P, &tmp1);
    
    // Ft = F'
    arm_mat_trans_f32(&F, &Ft);
    
    // P = tmp1 * Ft = F * P * F'
    arm_mat_mult_f32(&tmp1, &Ft, &ekf->P);
    
    // P = P + Q
    arm_mat_add_f32(&ekf->P, &ekf->Q, &ekf->P);
    
    // 5. Predict State Error: x = F * x
    float x_new_data[15];
    arm_matrix_instance_f32 x_new;
    arm_mat_init_f32(&x_new, 15, 1, x_new_data);
    
    arm_mat_mult_f32(&F, &ekf->x, &x_new);
    memcpy(ekf->x_data, x_new_data, 15 * sizeof(float));
}

void EKF_Update(EKF_Context_t *ekf, float lat, float lon, float alt) {
    // 1. Calculate Measurement Residual z
    float gps_enu[3];
    LLA_to_ENU(lat, lon, alt, ekf->lla0[0], ekf->lla0[1], ekf->lla0[2], gps_enu);
    
    float z_data[3];
    z_data[0] = gps_enu[0] - ekf->pos_enu[0];
    z_data[1] = gps_enu[1] - ekf->pos_enu[1];
    z_data[2] = gps_enu[2] - ekf->pos_enu[2];
    
    arm_matrix_instance_f32 z;
    arm_mat_init_f32(&z, 3, 1, z_data);

    // 2. H Matrix (3x15) - Identity on top left
    // S = H P H' + R
    // Since H is [I 0], H P H' is just the top-left 3x3 block of P.
    
    float S_data[9];
    arm_matrix_instance_f32 S;
    arm_mat_init_f32(&S, 3, 3, S_data);
    
    for(int r=0; r<3; r++) {
        for(int c=0; c<3; c++) {
            S_data[r*3 + c] = ekf->P_data[r*15 + c] + ekf->R_data[r*3 + c];
        }
    }

    // 3. Calculate Kalman Gain K = P H' S^-1
    // K = P[0:15, 0:3] * S^-1
    float S_inv_data[9];
    arm_matrix_instance_f32 S_inv;
    arm_mat_init_f32(&S_inv, 3, 3, S_inv_data);
    
    if (arm_mat_inverse_f32(&S, &S_inv) != 0) { // ARM_MATH_SUCCESS
        return; // Singularity
    }

    // Extract P * H' (which is first 3 columns of P)
    float PHt_data[15*3];
    arm_matrix_instance_f32 PHt;
    arm_mat_init_f32(&PHt, 15, 3, PHt_data);
    
    for(int r=0; r<15; r++) {
        for(int c=0; c<3; c++) {
            PHt_data[r*3 + c] = ekf->P_data[r*15 + c];
        }
    }
    
    float K_data[15*3];
    arm_matrix_instance_f32 K;
    arm_mat_init_f32(&K, 15, 3, K_data);
    
    arm_mat_mult_f32(&PHt, &S_inv, &K);

    // 4. Update State x = x + K * (z - H*x)
    // H*x is just x[0:3]
    float Hx_data[3];
    Hx_data[0] = ekf->x_data[0];
    Hx_data[1] = ekf->x_data[1];
    Hx_data[2] = ekf->x_data[2];
    
    float innov_data[3];
    innov_data[0] = z_data[0] - Hx_data[0];
    innov_data[1] = z_data[1] - Hx_data[1];
    innov_data[2] = z_data[2] - Hx_data[2];
    
    arm_matrix_instance_f32 innov;
    arm_mat_init_f32(&innov, 3, 1, innov_data);
    
    float K_innov_data[15];
    arm_matrix_instance_f32 K_innov;
    arm_mat_init_f32(&K_innov, 15, 1, K_innov_data);
    
    arm_mat_mult_f32(&K, &innov, &K_innov);
    arm_mat_add_f32(&ekf->x, &K_innov, &ekf->x);

    // 5. Update Covariance P = (I - K H) P
    // P = P - K * (H * P)
    // H * P is top 3 rows of P
    float HP_data[3*15];
    arm_matrix_instance_f32 HP;
    arm_mat_init_f32(&HP, 3, 15, HP_data);
    for(int r=0; r<3; r++) {
        for(int c=0; c<15; c++) {
            HP_data[r*15 + c] = ekf->P_data[r*15 + c];
        }
    }
    
    float KHP_data[15*15];
    arm_matrix_instance_f32 KHP;
    arm_mat_init_f32(&KHP, 15, 15, KHP_data);
    
    arm_mat_mult_f32(&K, &HP, &KHP);
    arm_mat_sub_f32(&ekf->P, &KHP, &ekf->P);

    // 6. Inject Error into Navigation State
    for(int i=0; i<3; i++) ekf->pos_enu[i] += ekf->x_data[0+i];
    for(int i=0; i<3; i++) ekf->vel_enu[i] += ekf->x_data[3+i];
    
    // Orientation injection
    // Python: ypr[0] += x[8] (Yaw), ypr[1] += x[7] (Pitch), ypr[2] += x[6] (Roll)
    ekf->orient_ypr[0] += ekf->x_data[8]; // Yaw
    ekf->orient_ypr[1] += ekf->x_data[7]; // Pitch
    ekf->orient_ypr[2] += ekf->x_data[6]; // Roll

    // 7. Reset Error State (only 0-9)
    for(int i=0; i<9; i++) ekf->x_data[i] = 0.0f;

    // Sync biases to struct members for external monitoring
    for(int i=0; i<3; i++) {
    	ekf->acc_bias[i] = ekf->x_data[9+i];
    	ekf->gyr_bias[i] = ekf->x_data[12+i];
    }
}
