function compare_ga_gap12()
    %% Load GAP12 dataset
    dataFile = 'gap12.txt';
    fileID = fopen(dataFile, 'r');
    if fileID == -1
        error('Cannot open file: %s', dataFile);
    end

    numInstances = fscanf(fileID, '%d', 1);  % number of instances in the dataset

    % Read only the first instance
    numServers = fscanf(fileID, '%d', 1);
    numUsers = fscanf(fileID, '%d', 1);
    costMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    resourceMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    capacityLimits = fscanf(fileID, '%f', [numServers, 1]);
    fclose(fileID);

    %% Run Binary Coded GA
    [~, bcga_convergence] = run_binary_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

    %% Run Real Coded GA
    [~, rcga_convergence] = run_real_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

    %% Plot Comparison
    figure;
    plot(bcga_convergence, 'LineWidth', 2);
    hold on;
    plot(rcga_convergence, 'LineWidth', 2);
    xlabel('Generation');
    ylabel('Best Fitness');
    legend('Binary Coded GA', 'Real Coded GA');
    title('GAP12 Convergence: Binary vs Real Coded GA');
    grid on;
end

%% ------------------- Binary GA -------------------
function [bestSol, convergence] = run_binary_ga(m, n, costMatrix, resourceMatrix, capacityLimits)
    popSize = 100;
    generations = 100;
    crossRate = 0.8;
    mutateRate = 0.02;

    pop = randi([0, 1], popSize, m * n);
    for i = 1:popSize
        pop(i,:) = enforce_binary_feasibility(pop(i,:), m, n);
    end

    fitness = evaluate_population(pop, m, n, costMatrix, resourceMatrix, capacityLimits);
    convergence = zeros(generations, 1);

    for gen = 1:generations
        parents = binary_tournament_selection(pop, fitness);
        offspring = single_point_crossover(parents, crossRate);
        offspring = bitflip_mutation(offspring, mutateRate);

        for i = 1:popSize
            offspring(i,:) = enforce_binary_feasibility(offspring(i,:), m, n);
        end

        offspringFitness = evaluate_population(offspring, m, n, costMatrix, resourceMatrix, capacityLimits);
        [pop, fitness] = elitism_selection([pop; offspring], [fitness; offspringFitness], popSize);

        convergence(gen) = max(fitness);
    end

    [~, bestIdx] = max(fitness);
    bestSol = reshape(pop(bestIdx, :), [m, n]);
end

function newSol = enforce_binary_feasibility(sol, m, n)
    mat = reshape(sol, [m, n]);
    for j = 1:n
        [~, idx] = max(mat(:, j));
        mat(:, j) = 0;
        mat(idx, j) = 1;
    end
    newSol = reshape(mat, [1, m * n]);
end

function fitness = evaluate_population(pop, m, n, costMatrix, resourceMatrix, capacityLimits)
    fitness = zeros(size(pop,1),1);
    for i = 1:size(pop,1)
        sol = reshape(pop(i,:), [m, n]);
        benefit = sum(sum(sol .* costMatrix));
        violation = sum(max(sum(sol .* resourceMatrix, 2) - capacityLimits, 0));
        assignmentError = sum(abs(sum(sol,1) - 1));
        penalty = 1e6 * (violation + assignmentError);
        fitness(i) = benefit - penalty;
    end
end

function selected = binary_tournament_selection(pop, fitness)
    popSize = size(pop,1);
    selected = zeros(size(pop));
    for i = 1:popSize
        a = randi(popSize);
        b = randi(popSize);
        if fitness(a) > fitness(b)
            selected(i,:) = pop(a,:);
        else
            selected(i,:) = pop(b,:);
        end
    end
end

function offspring = single_point_crossover(parents, rate)
    [popSize, numGenes] = size(parents);
    offspring = parents;
    for i = 1:2:popSize-1
        if rand < rate
            point = randi(numGenes-1);
            offspring(i,point+1:end) = parents(i+1,point+1:end);
            offspring(i+1,point+1:end) = parents(i,point+1:end);
        end
    end
end

function mutated = bitflip_mutation(pop, rate)
    mutated = pop;
    for i = 1:numel(pop)
        if rand < rate
            mutated(i) = 1 - pop(i);
        end
    end
end

function [newPop, newFit] = elitism_selection(pop, fit, maxSize)
    [sortedFit, idx] = sort(fit, 'descend');
    newPop = pop(idx(1:maxSize), :);
    newFit = sortedFit(1:maxSize);
end

%% ------------------- Real Coded GA -------------------
function [bestSol, convergence] = run_real_ga(m, n, costMatrix, resourceMatrix, capacityLimits)
    popSize = 100;
    generations = 100;
    crossRate = 0.9;
    mutateRate = 0.05;

    pop = rand(popSize, m * n);
    for i = 1:popSize
        pop(i,:) = enforce_real_feasibility(pop(i,:), m, n);
    end

    fitness = evaluate_population(pop, m, n, costMatrix, resourceMatrix, capacityLimits);
    convergence = zeros(generations, 1);

    for gen = 1:generations
        parents = binary_tournament_selection(pop, fitness);
        offspring = sbx_crossover(parents, crossRate);
        offspring = gaussian_mutation(offspring, mutateRate);

        for i = 1:popSize
            offspring(i,:) = enforce_real_feasibility(offspring(i,:), m, n);
        end

        offspringFitness = evaluate_population(offspring, m, n, costMatrix, resourceMatrix, capacityLimits);
        [pop, fitness] = elitism_selection([pop; offspring], [fitness; offspringFitness], popSize);

        convergence(gen) = max(fitness);
    end

    [~, bestIdx] = max(fitness);
    bestSol = reshape(pop(bestIdx, :), [m, n]);
end

function newSol = enforce_real_feasibility(sol, m, n)
    mat = reshape(sol, [m, n]);
    for j = 1:n
        mat(:,j) = mat(:,j) / sum(mat(:,j) + eps); % normalize
    end
    newSol = reshape(mat, [1, m * n]);
end

function offspring = sbx_crossover(parents, rate)
    [popSize, numGenes] = size(parents);
    offspring = parents;
    eta = 15;

    for i = 1:2:popSize-1
        if rand < rate
            u = rand(1, numGenes);
            beta = zeros(1, numGenes);
            mask = u <= 0.5;
            beta(mask) = (2 * u(mask)).^(1 / (eta + 1));
            beta(~mask) = (1 ./ (2 * (1 - u(~mask)))).^(1 / (eta + 1));

            offspring(i,:) = 0.5 * ((1 + beta) .* parents(i,:) + (1 - beta) .* parents(i+1,:));
            offspring(i+1,:) = 0.5 * ((1 - beta) .* parents(i,:) + (1 + beta) .* parents(i+1,:));
        end
    end
end

function mutated = gaussian_mutation(pop, rate)
    [popSize, numGenes] = size(pop);
    mutated = pop + rate * randn(popSize, numGenes);
    mutated = min(max(mutated, 0), 1); % clip to [0,1]
end
