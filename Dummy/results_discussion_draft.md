# 4. Results and Discussion

This section presents the comprehensive results of the experimental evaluation across 354 GA operator combinations on the 56 Solomon VRPTW instances. The analysis focuses on three key dimensions: the impact of representation schemes, the interaction between genetic operators, and the robustness of the top-performing configurations across different problem classes.

## 4.1 Global Performance Overview and Impact of Representation

The experimental results unequivocally demonstrate the superiority of the permutation-based representation for the Vehicle Routing Problem. Figure 2 illustrates the distribution of the Fitness Gap (normalized deviation from the best-known solution) for each representation scheme across the three problem classes (C, R, and RC).

**[Insert Figure 2 here: Boxplots of Fitness Gap by Representation]**

As shown in **Figure 2**, permutation-based methods (green) consistently achieve significantly lower fitness gaps compared to real-valued (orange) and binary (blue) representations. This trend holds across all problem types:
*   **Clustered (C) Instances**: Permutation methods achieve near-optimal performance with very low variance, while binary and real-valued approaches struggle to find feasible low-cost solutions, resulting in large fitness gaps.
*   **Random (R) and Random-Clustered (RC) Instances**: The gap widens further, highlighting that the direct encoding of customer sequences in permutation representations is far more effective at preserving route validity and geometrical structure than the indirect decoding required by continuous or binary encodings.

The dominance of permutation encodings supports the hypothesis that preserving the discrete sequential nature of the VRP is critical for heuristic effectiveness.

## 4.2 Operator Interaction Analysis

To pinpoint the specific contributions of Selection, Crossover, and Mutation strategies, we utilized 3D scatter plots to visualize the fitness landscape of each representation scheme. **Figure 4** presents these design space analyses.

**[Insert Figure 4 here: 3D Scatter Plots of Operator Interactions]**

*   **Permutation Results (Fig 4a)**: The analysis reveals a distinct "valley of optimality" associated with specific operator combinations. Notably, **Sequential Constructive Crossover (SCX)** paired with **Inversion** or **Scramble** mutation yields the highest-quality solutions. This suggests that operators explicitly designed to construct distance-minimized edges (like SCX) synergize well with mutation operators that perform local reordering (Inversion), effectively balancing exploration and exploitation.
*   **Binary and Real-Valued Results (Fig 4b, 4c)**: These representations show a more scattered landscape with fewer high-performance clusters. For real-valued representations, **SBX** crossover combined with **Polynomial** mutation (a standard in continuous optimization) offers the best relative performance, though it still lags behind permutation baselines. This indicates that while "standard" continuous GA operators function correctly, they are less efficient at navigating the specific combinatorial constraints of the VRP.

## 4.3 Method Robustness and Stability

The robustness of the top-performing algorithms was evaluated by examining their performance variance across individual benchmark instances. **Figure 6** provides a heatmap of the normalized performance for the top 15 method configurations.

**[Insert Figure 6 here: Heatmap of Top 15 Methods]**

The heatmap reveals several critical insights:
1.  **Consistent Leaders**: The top-ranked combination, **Permutation-Rank-SCX-Inversion**, displays a remarkable "green streak" across most instances, indicating high stability regardless of customer distribution (Random vs. Clustered).
2.  **Problem Class Sensitivity**: Some methods perform excellently on Clustered (C1, C2) instances but degrade on Random (R1, R2) instances. However, the top SCX-based variants maintain competitive fitness levels even on the harder R and RC classes.
3.  **Selection Pressure**: The presence of **Rank** and **Tournament** selection in the top tier confirms that controlling selection pressure is vital to prevent premature convergence while ensuring sufficient evolutionary drive.

## 4.4 Summary of Findings

In summary, the extensive comparative study of 354 operator combinations identifies **Permutation Representation** as the fundamental requirement for high-performance VRP solvers. Within this class, the **Sequential Constructive Crossover (SCX)** proves to be a standout operator, likely due to its heuristic awareness of edge costs. When coupled with **Inversion Mutation** and a diversity-preserving selection scheme like **Rank Selection**, the GA achieves robust, high-quality solutions that significantly outperform standard binary or real-valued genetic algorithm implementations.
