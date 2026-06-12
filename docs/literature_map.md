        # Literature Map

        Paper: 93 counterfactual_affordance_maps

        Field box: robot affordance learning

        Thesis: Counterfactual Affordance Maps turns the seed bet into a mechanism: Map affordances that would exist under alternate grasps, poses, and supports.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching (2019)
- Interactive affordance map building for a robotic task (2015)
- To Afford or Not to Afford: A New Formalization of Affordances Toward Affordance-Based Robot Control (2007)
- A Real-Time Semantic Map Production System for Indoor Robot Navigation (2024)
- Certifiably optimal rotation and pose estimation based on the Cayley map (2025)
- Hybrid Map-Based Path Planning for Robot Navigation in Unstructured Environments (2023)
- SENT Map -- Semantically Enhanced Topological Maps with Foundation Models (2025)
- Development of simulation software for mobile robot path planning within multilayer map system based on metric and topological maps (2017)
- Occupancy-SLAM: An Efficient and Robust Algorithm for Simultaneously Optimizing Robot Poses and Occupancy Map (2025)
- Merging Occupancy Grid Maps From Multiple Robots (2006)
- GAT-Grasp: Gesture-Driven Affordance Transfer for Task-Aware Robotic Grasping (2025)
- MIRA: Mental Imagery for Robotic Affordances (2022)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Counterfactual affordance maps keeps action-critical alternatives explicit until a physical observation collapses them.
