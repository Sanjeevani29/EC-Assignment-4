   
function compare_ga_gap12()
    %% Load GAP12 dataset
    dataFile = 'gap12.txt';
    fileID = fopen(dataFile, 'r');
    if fileID == -1
        error('Cannot open file: %s', dataFile);
    end

    numInstances = fscanf(fileID, '%d', 1);  % number of instances

    % Read only the first instance
    numServers = fscanf(fileID, '%d', 1);
    numUsers = fscanf(fileID, '%d', 1);
    costMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    resourceMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    capacityLimits = fscanf(fileID, '%f', [numServers, 1]);
    fclose(fileID);

    %% Run Binary GA
    disp("Running Binary Coded GA...");
    [~, bcga_convergence] = run_binary_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

    %% Run Real Coded GA
    disp("Running Real Coded GA...");
    [~, rcga_convergence] = run_real_ga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);

    %% Plot Comparison
    figure;
    plot(bcga_convergence, 'r-', 'LineWidth', 2);
    hold on;
    plot(rcga_convergence, 'b--', 'LineWidth', 2);
    xlabel('Generation');
    ylabel('Best Fitness');
    legend('Binary Coded GA', 'Real Coded GA');
    title('GAP12 Convergence: Binary vs Real Coded GA');
    grid on;
end

%% ------------------- Binary GA -------------------
function [bestSol, convergence] = run_binary_ga(m, n, costMatrix, resourceMatrix, capacityLimits)
    popSize = 100; generations = 100;
    crossRate = 0.9; mutateRate = 0.02;
    convergence = zeros(generations, 1);

    % Initialization
    pop = zeros(popSize, m*n);
    for i = 1:popSize
        sol = zeros(m, n);
        for j = 1:n
            sol(randi(m), j) = 1;
        end
        pop(i,:) = reshape(sol, 1, []);
    end

    fitness = evaluate_population(pop, m, n, costMatrix, resourceMatrix, capacityLimits);

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

%% ------------------- Real GA -------------------
function [bestSol, convergence] = run_real_ga(m, n, costMatrix, resourceMatrix, capacityLimits)
    popSize = 100; generations = 100;
    crossRate = 0.9; mutateRate = 0.05;
    convergence = zeros(generations, 1);

    % Initialization
    pop = rand(popSize, m*n);

    fitness = zeros(popSize,1);
    for i = 1:popSize
        decoded = decode_real_solution(pop(i,:), m, n);
        fitness(i) = compute_fitness(decoded, costMatrix, resourceMatrix, capacityLimits);
    end

    for gen = 1:generations
        parents = binary_tournament_selection(pop, fitness);
        offspring = sbx_crossover(parents, crossRate);
        offspring = gaussian_mutation(offspring, mutateRate);

        % Decode and evaluate offspring
        offspringFitness = zeros(popSize,1);
        for i = 1:popSize
            decoded = decode_real_solution(offspring(i,:), m, n);
            offspringFitness(i) = compute_fitness(decoded, costMatrix, resourceMatrix, capacityLimits);
        end

        [pop, fitness] = elitism_selection([pop; offspring], [fitness; offspringFitness], popSize);
        convergence(gen) = max(fitness);
    end

    [~, bestIdx] = max(fitness);
    bestSol = decode_real_solution(pop(bestIdx, :), m, n);
end

function mat = decode_real_solution(sol, m, n)
    mat = reshape(sol, [m, n]);
    binMat = zeros(m, n);
    for j = 1:n
        [~, idx] = max(mat(:, j));
        binMat(idx, j) = 1;
    end
    mat = binMat;
end

function fit = compute_fitness(sol, costMatrix, resourceMatrix, capacityLimits)
    benefit = sum(sum(sol .* costMatrix));
    violation = sum(max(sum(sol .* resourceMatrix, 2) - capacityLimits, 0));
    assignmentError = sum(abs(sum(sol,1) - 1));
    penalty = 1e6 * (violation + assignmentError);
    fit = benefit - penalty;
end

%% ------------------- Shared GA Functions -------------------
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
            temp1 = parents(i,:);
            temp2 = parents(i+1,:);
            offspring(i,:) = [temp1(1:point), temp2(point+1:end)];
            offspring(i+1,:) = [temp2(1:point), temp1(point+1:end)];
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

function offspring = sbx_crossover(parents, rate)
    [popSize, numGenes] = size(parents);
    offspring = parents;
    eta = 15;

    for i = 1:2:popSize-1
        if rand < rate
            u = rand(1, numGenes);
            beta = zeros(1, numGenes);
            for j = 1:numGenes
                if u(j) <= 0.5
                    beta(j) = (2 * u(j))^(1 / (eta + 1));
                else
                    beta(j) = (1 / (2 * (1 - u(j))))^(1 / (eta + 1));
                end
            end
            parent1 = parents(i,:);
            parent2 = parents(i+1,:);
            offspring(i,:) = 0.5 * ((1 + beta) .* parent1 + (1 - beta) .* parent2);
            offspring(i+1,:) = 0.5 * ((1 - beta) .* parent1 + (1 + beta) .* parent2);
        end
    end
end

function mutated = gaussian_mutation(pop, rate)
    [popSize, numGenes] = size(pop);
    mutated = pop + rate * randn(popSize, numGenes);
    mutated = min(max(mutated, 0), 1); % clamp values to [0,1]
end

function [newPop, newFit] = elitism_selection(pop, fit, maxSize)
    [sortedFit, idx] = sort(fit, 'descend');
    newPop = pop(idx(1:maxSize), :);
    newFit = sortedFit(1:maxSize);
end

