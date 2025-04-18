function optimize_gap_rcga()
    outputFile = 'realcodedGA_output.txt';
    fid_out = fopen(outputFile, 'w'); % Create/overwrite output file
    
    if fid_out == -1
        error('Cannot create output file: %s', outputFile);
    end

    for datasetIdx = 1:12
        dataFile = sprintf('gap%d.txt', datasetIdx);
        fileID = fopen(dataFile, 'r');
        if fileID == -1
            error('Error opening file %s.', dataFile);
        end

        numInstances = fscanf(fileID, '%d', 1);
        datasetName = sprintf('%s', dataFile(1:end-4));
        fprintf('\n%s\n', datasetName);
        fprintf(fid_out, '\n%s\n', datasetName);

        for instanceIdx = 1:numInstances
            numServers = fscanf(fileID, '%d', 1);
            numUsers = fscanf(fileID, '%d', 1);
            costMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
            resourceMatrix = fscanf(fileID, '%f', [numUsers, numServers])';
            capacityLimits = fscanf(fileID, '%f', [numServers, 1]);

            assignmentMatrix = execute_rcga(numServers, numUsers, costMatrix, resourceMatrix, capacityLimits);
            totalBenefit = sum(sum(costMatrix .* assignmentMatrix));

            resultLine = sprintf('c%d-%d  %d\n', numServers*100 + numUsers, instanceIdx, round(totalBenefit));
            fprintf(resultLine);
            fprintf(fid_out, resultLine);
        end
        fclose(fileID);
    end

    fclose(fid_out);
    fprintf('\nAll GAP instance results saved to "%s"\n', outputFile);
end

% --- Remaining functions stay same as in your code, re-attached for completeness ---

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

function score = compute_fitness(solution, costMatrix, resourceMatrix, capacityLimits, numServers, numUsers)
    reshapedSolution = reshape(solution, [numServers, numUsers]);
    totalCost = sum(sum(costMatrix .* reshapedSolution));
    capacityExceedance = sum(max(sum(reshapedSolution .* resourceMatrix, 2) - capacityLimits, 0));
    incorrectAssignment = sum(abs(sum(reshapedSolution, 1) - 1));
    penaltyFactor = 1e6 * (capacityExceedance + incorrectAssignment);
    score = totalCost - penaltyFactor;
end

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

function mutated = apply_gaussian_mutation(offspring, mutationProb)
    [popSize, numGenes] = size(offspring);
    mutated = offspring + mutationProb * randn(popSize, numGenes);
    mutated = max(0, min(1, mutated));
end

function [newPopulation, newFitness] = elitism_selection(oldPop, oldFit, newPop, newFit)
    combinedPop = [oldPop; newPop];
    combinedFit = [oldFit, newFit];
    [~, sortedIdx] = sort(combinedFit, 'descend');
    newPopulation = combinedPop(sortedIdx(1:size(oldPop, 1)), :);
    newFitness = combinedFit(sortedIdx(1:size(oldPop, 1)));
end

function feasibleSolution = adjust_feasibility(solution, numServers, numUsers)
    reshapedSol = reshape(solution, [numServers, numUsers]);
    for j = 1:numUsers
        reshapedSol(:, j) = reshapedSol(:, j) / sum(reshapedSol(:, j));
    end
    feasibleSolution = reshape(reshapedSol, [1, numServers * numUsers]);
end

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

