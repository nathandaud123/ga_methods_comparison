# 4. Results and Discussion

This section presents a comprehensive evaluation of 354 Genetic Algorithm operator combinations applied to the 56 Solomon test instances. The results are analyzed across three hierarchical levels: the fundamental impact of representation schemes, the specific performance of genetic operators (selection, crossover, mutation), and the robustness of the top-performing configurations. All fitness values reported refer to the total distance traveled, where lower values indicate better performance.

## 4.1 Global Performance Overview and Impact of Representation

The choice of representation scheme proved to be the most critical determinant of algorithm performance. **Table 5a** and **Figure 2** summarize the aggregate performance across the three representation types.

**[Insert Table 5a here: Breakdown of Representation Performance]**

The **Permutation** representation significantly outperformed both binary and real-valued encodings, achieving an average fitness of **2094.37** compared to **2861.02** for real-valued and **2868.07** for binary representations. This performance gap is further visualized in **Figure 2**, where permutation-based methods (green boxplots) consistently maintain lower fitness gaps across all problem classes (Clustered, Random, and Random-Clustered).

**[Insert Figure 2 here: Boxplots of Fitness Gap by Representation]**

This superiority can be attributed to the combinatorial nature of the VRP. Permutation encodings directly map to the sequence of customer visits, allowing operators to preserve valid route structures and geometric relationships. In contrast, binary and real-valued representations rely on indirect decoding mechanisms (e.g., sorting random keys), which often disrupt the "building blocks" of good solutions during crossover, leading to slower convergence and lower-quality final solutions.

## 4.2 Operator Analysis and Interaction Effects

Beyond representation, the interaction between specific genetic operators played a pivotal role in shaping the search landscape. **Figure 4** presents a 3D design space analysis illustrating these interactions.

**[Insert Figure 4 here: 3D Scatter Plots of Operator Interactions]**

### 4.2.1 Selection Strategies
While **Elitism** yielded the best *average* performance (Mean Fitness: 1610.53) across all permutation experiments (**Table 5b**), the absolute best solutions were found using **Rank** and **Tournament** selection when paired with specific aggressive operators.

**[Insert Table 5b here: Breakdown of Selection Performance]**

For instance, although Rank selection had a higher average fitness (2316.71) due to poor performance with weak crossovers, it was a key component of the top-performing combination (Rank + SCX + Inversion). This suggests that while Elitism provides a safe baseline, pressure-based schemes like Rank selection are necessary to drive the population toward the global optimum when powerful local search operators are present.

### 4.2.2 Crossover Operators
The impact of crossover was decisive. Within the permutation domain, the **Sequential Constructive Crossover (SCX)** demonstrated overwhelming superiority, achieving an average fitness of **1169.05** (**Table 5c**), which is nearly 50% better than standard operators like CX (2057.47) and PMX (2141.86).

**[Insert Table 5c here: Breakdown of Crossover Performance]**

SCX explicitly utilizes distance information to construct offspring, effectively acting as a heuristic crossover that preserves low-cost edges. In contrast, "blind" operators like PMX or OX, which focus only on order or position without geometric context, struggled to find high-quality routes in the complex landscape of the Solomon instances.

### 4.2.3 Mutation Operators
Mutation analysis reveals an interesting interaction effect. On aggregate (**Table 5d**), **Swap** mutation appeared most robust (Mean Fitness: 1855.32). However, the global best solution utilized **Inversion** mutation.

**[Insert Table 5d here: Breakdown of Mutation Performance]**

Inversion acts as a 2-opt local search move, reversing route segments to untangle crossing paths. While risky in early generations (hence the higher average fitness of 2126.30), it provides the necessary local refinement to polish the high-quality skeletons produced by SCX. This synergy between a constructive crossover (SCX) and a local-improvement mutation (Inversion) was the defining characteristic of the top-tier algorithms.

## 4.3 Method Robustness and Stability

The stability of the top 15 discovered method configurations was evaluated across the diverse problem structures of the Solomon benchmark (C, R, and RC types). **Figure 6** illustrates the normalized performance heatmap.

**[Insert Figure 6 here: Heatmap of Top 15 Methods]**

The heatmap confirms the **Permutation-Rank-SCX-Inversion** combination as the most robust solver, displaying a "green streak" of near-optimal performance across both Clustered (C) and Random (R) instances. Notably, methods relying on random Crossover (e.g., OX, PMX) often degraded significantly on Random (R) instances, whereas SCX-based methods maintained their performance advantage, demonstrating that heuristic-guided recombination is essential for handling unstructured customer distributions.

## 4.4 Conclusion

The extensive empirical evaluation confirms that the optimal configuration for the Vehicle Routing Problem involves a **Permutation** representation, utilizing **Sequential Constructive Crossover (SCX)** for global structural search, paired with **Inversion** mutation for local refinement, and driven by **Rank** or **Tournament** selection to prevent stagnation. This specific "recipe" outperforms standard genetic algorithm implementations by a wide margin, offering a robust framework for solving complex routing challenges.
