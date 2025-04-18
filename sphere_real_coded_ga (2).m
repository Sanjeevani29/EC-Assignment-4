function run_rcga_sphere()
    % Sphere function settings
    dim = 30;                      % Dimensionality
    bounds = [-10, 10];            % Search space
    populationSize = 100;
    generations = 300;
    crossoverProb = 0.9;
    mutationProb = 0.05;

    % Initialize population
    population = bounds(1) + (bounds(2) - bounds(1)) * rand(populationSize, dim);
    fitnessHistory = zeros(generations, 1);

    % Evaluate initial fitness
    fitnessScores = arrayfun(@(i) -sphere_function(population(i, :)), 1:populationSize);

    for gen = 1:generations
        selectedParents = perform_selection(population, fitnessScores);
        offspring = apply_sbx(selectedParents, crossoverProb, bounds);
        mutatedOffspring = apply_gaussian_mutation(offspring, mutationProb, bounds);

        newFitness = arrayfun(@(i) -sphere_function(mutatedOffspring(i, :)), 1:populationSize);
        [population, fitnessScores] = elitism_selection(population, fitnessScores, mutatedOffspring, newFitness);

        bestFitness = max(fitnessScores);           % Since we minimize, we store -fitness
        fitnessHistory(gen) = -bestFitness;         % Convert back to positive for plotting
    end

    % Plot convergence graph
    figure;
    plot(1:generations, fitnessHistory, 'b-', 'LineWidth', 2);
    xlabel('Generation');
    ylabel('Best Fitness Value (Sphere)');
    title('Convergence Curve - RCGA on Sphere Function');
    grid on;
end

function f = sphere_function(x)
    f = sum(x .^ 2);
end

function offspring = apply_sbx(parents, crossoverProb, bounds)
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
    
    % Ensure bounds
    offspring = min(bounds(2), max(bounds(1), offspring));
end

function mutated = apply_gaussian_mutation(offspring, mutationProb, bounds)
    [popSize, numGenes] = size(offspring);
    mutated = offspring + mutationProb * randn(popSize, numGenes);
    mutated = max(bounds(1), min(bounds(2), mutated));
end

function [newPop, newFit] = elitism_selection(oldPop, oldFit, newPop, newFit)
    combinedPop = [oldPop; newPop];
    combinedFit = [oldFit, newFit];
    [~, sortedIdx] = sort(combinedFit, 'descend'); % Maximize fitness
    newPop = combinedPop(sortedIdx(1:size(oldPop, 1)), :);
    newFit = combinedFit(sortedIdx(1:size(oldPop, 1)));
end

function selected = perform_selection(population, fitnessValues)
    popSize = size(population, 1);
    selected = zeros(size(population));
    for i = 1:popSize
        idx1 = randi(popSize);
        idx2 = randi(popSize);
        if fitnessValues(idx1) > fitnessValues(idx2)
            selected(i, :) = population(idx1, :);
        else
            selected(i, :) = population(idx2, :);
        end
    end
end


