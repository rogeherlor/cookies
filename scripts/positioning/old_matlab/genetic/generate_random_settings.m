function [settings] = generate_random_settings(n_settings,settings)
    
    % Se itera con las 5 mejores combinaciones de parametros
    n_keep = 5;
    mutations = n_settings/n_keep;
    
    for k = 1:n_keep
        for i = 1:(mutations-1)
            % Indice donde se guardan los nuevos valores
            m = k+(n_keep*i);
            
            % accBias. Result > 0
            accBias = abs(settings(k).accBias + normrnd(0,1e-7));
            settings(m).accBias = accBias;

            % gyrBias. Result > 0
            gyrBias = abs(settings(k).gyrBias + normrnd(0,1e-7));
            settings(m).gyrBias = gyrBias;

            % beta_acc. Result > 0
            beta_acc = abs(settings(k).beta_acc + normrnd(0,1e-1));
            settings(m).beta_acc = beta_acc;

            % beta_gyr. Result > 0
            beta_gyr = abs(settings(k).beta_gyr + normrnd(0,1e-1));
            settings(m).beta_gyr = beta_gyr;

            % Rpos. Result > 0
            Rpos = abs(settings(k).Rpos + normrnd(0,1e-1));
            settings(m).Rpos = Rpos;

            % Qpos. Result > 0
            Qpos= abs(settings(k).Qpos + normrnd(0,1e-1));
            settings(m).Qpos = Qpos;

            % Qvel. Result > 0
            Qvel= abs(settings(k).Qvel + normrnd(0,1e-1));
            settings(m).Qvel = Qvel;
            
            % Qorient. Result > 0
            Qorient= abs(settings(k).Qorient + normrnd(0,1e-1));
            settings(m).Qorient = Qorient;

            % Qacc. Result > 0
            Qacc= abs(settings(k).Qacc + normrnd(0,1e-1));
            settings(m).Qacc = Qacc;

            % Qgyr. Result > 0
            Qgyr= abs(settings(k).Qgyr + normrnd(0,1e-1));
            settings(m).Qgyr = Qgyr;
           
        end
    end

end