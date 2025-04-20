function optimize_gap12_instance1_avg20()
    datasetIdx = 12;  % Only GAP12
    dataFile = sprintf('gap%d.txt', datasetIdx);
    fileID = fopen(dataFile, 'r');
    if fileID == -1
        error('Error opening file %s.', dataFile);
    end

    numInstances = fscanf(fileID, '%d', 1);
    if numInstances < 1
        error('GAP12 must have at least 1 instance.');
    end

    fprintf('\nRunning GAP12 - Instance 1 for 20 iterations\n');

    % Read only the 1st instance
    numServers = fscanf(fileID, '%d', 1);
    numUsers = fscanf(fileID, '%d', 1);
    costMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    resourceMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
    capacityLimits = fscanf(fileID, '%f', [numServers, 1]);
    fclose(fileID);

    totalBenefits = zeros(1, 20);

    for run = 1:20
        assignmentMatrix = execute_rcga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);
        assignmentMatrix = round(assignmentMatrix);  % Ensure binary

        totalBenefit = sum(sum(costMatrix .* assignmentMatrix));
        totalBenefits(run) = totalBenefit;
        fprintf('Run %2d: Total Benefit = %d\n', run, round(totalBenefit));
    end

    averageBenefit = mean(totalBenefits);
    fprintf('\nAverage Total Benefit after 20 runs (GAP12 - Instance 1): %.2f\n', averageBenefit);
end

% --- Real-Coded Genetic Algorithm ---
function assignmentMatrix = execute_rcga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits)
    populationSize = 100;
    generations = 300;
    crossoverProb = 0.9;
    mutationProb = 0.05;

    population = rand(populationSize, numServers * numUsers);
    
    for idx = 1:populationSize
        population(idx, :) = adjust_feasibility(population(idx, :), numServers, numUsers);
    end

    fitnessScores = arrayfun(@(i) compute_fitness(population(i, :), costMatrix, resourceMatrix, capacityLimits, numServers, numUsers), 1:populationSize);

    for gen = 1:generations
        selectedParents = perform_selection(population, fitnessScores);
        offspring = apply_sbx(selectedParents, crossoverProb);
        mutatedOffspring = apply_gaussian_mutation(offspring, mutationProb);

        for i = 1:size(mutatedOffspring, 1)
            mutatedOffspring(i, :) = adjust_feasibility(mutatedOffspring(i, :), numServers, numUsers);
        end

        newFitness = arrayfun(@(i) compute_fitness(mutatedOffspring(i, :), costMatrix, resourceMatrix, capacityLimits, numServers, numUsers), 1:size(mutatedOffspring, 1));

        [population, fitnessScores] = elitism_selection(population, fitnessScores, mutatedOffspring, newFitness);
    end

    [~, bestIdx] = max(fitnessScores);
    assignmentMatrix = reshape(population(bestIdx, :), [numServers, numUsers]);
end

% --- Fitness Evaluation ---
function score = compute_fitness(solution, costMatrix, resourceMatrix, capacityLimits, numServers, numUsers)
    reshapedSolution = reshape(solution, [numServers, numUsers]);
    totalCost = sum(sum(costMatrix .* reshapedSolution));

    capacityExceedance = sum(max(sum(reshapedSolution .* resourceMatrix, 2) - capacityLimits, 0));
    incorrectAssignment = sum(abs(sum(reshapedSolution, 1) - 1));
    penaltyFactor = 1e6 * (capacityExceedance + incorrectAssignment);

    score = totalCost - penaltyFactor;
end

% --- SBX Crossover ---
function offspring = apply_sbx(parents, crossoverProb)
    [popSize, numGenes] = size(parents);
    offspring = parents;

    for i = 1:2:popSize-1
        if rand < crossoverProb
            u = rand(1, numGenes);
            beta = (2 .* u) .^ (1/3) .* (u <= 0.5) + (1 ./ (2 .* (1 - u))) .^ (1/3) .* (u > 0.5);

            offspring(i, :) = 0.5 * ((1 + beta) .* parents(i, :) + (1 - beta) .* parents(i+1, :));
            offspring(i+1, :) = 0.5 * ((1 - beta) .* parents(i, :) + (1 + beta) .* parents(i+1, :));
        end
    end
end

% --- Gaussian Mutation ---
function mutated = apply_gaussian_mutation(offspring, mutationProb)
    [popSize, numGenes] = size(offspring);
    mutated = offspring + mutationProb * randn(popSize, numGenes);
    mutated = max(0, min(1, mutated));  % Ensure values in [0,1]
end

% --- Elitism Selection ---
function [newPopulation, newFitness] = elitism_selection(oldPop, oldFit, newPop, newFit)
    combinedPop = [oldPop; newPop];
    combinedFit = [oldFit, newFit];

    [~, sortedIdx] = sort(combinedFit, 'descend');
    newPopulation = combinedPop(sortedIdx(1:size(oldPop, 1)), :);
    newFitness = combinedFit(sortedIdx(1:size(oldPop, 1)));
end

% --- Feasibility Adjustment: Assign each user to exactly 1 server ---
function feasibleSolution = adjust_feasibility(solution, numServers, numUsers)
    reshapedSol = reshape(solution, [numServers, numUsers]);

    for j = 1:numUsers
        [~, maxIdx] = max(reshapedSol(:, j));
        reshapedSol(:, j) = 0;
        reshapedSol(maxIdx, j) = 1;
    end

    feasibleSolution = reshape(reshapedSol, [1, numServers * numUsers]);
end

% --- Tournament Selection ---
function selectedParents = perform_selection(population, fitnessValues)
    popSize = size(population, 1);
    selectedParents = zeros(size(population));

    for i = 1:popSize
        idx1 = randi(popSize);
        idx2 = randi(popSize);

        if fitnessValues(idx1) > fitnessValues(idx2)
            selectedParents(i, :) = population(idx1, :);
        else
            selectedParents(i, :) = population(idx2, :);
        end
    end
end
