text = """
Autonomous driving decision-making has historically relied on a broad range of paradigms. Rule-based systems~\cite{bouchard2015rulebased} define rigid behavior policies, and finite state machines~\cite{bae2020fsm} model driving states explicitly; these offer high interpretability but often scale poorly as environmental complexity increases due to the impossibility of anticipating every driving scenario. To overcome these limitations, machine learning approaches, particularly imitation learning~\cite{codevilla2018il,cheng2023pluto,moncalvillo2024ackermann}, have been widely adopted to learn end-to-end control or planning from expert demonstrations. While these models can achieve strong performance under nominal conditions, they frequently struggle with generalization when encountering out-of-distribution events or safety-critical edge cases not present in the training dataset.

To address the challenges of generalization and long-term planning under uncertainty, reinforcement learning, and specifically deep reinforcement learning, has emerged as a central paradigm. By leveraging deep neural networks to approximate policies and value functions, deep reinforcement learning enables effective learning from rich sensory inputs and explicit optimization of long-term behavior. Recent surveys by Zhao et al.~\cite{zhao2024survey} and Wu et al.~\cite{wu2024survey} detail the rapid adoption of deep reinforcement learning across various autonomous driving domains, noting its capacity to handle high-dimensional continuous state and action spaces. Concurrently, several benchmark environments, such as TorchDriveEnv~\cite{lavington2024torchdriveenv} and specialized intersection suites~\cite{liu2021rlbenchmark}, have been proposed to standardize the evaluation of these agents. As documented in other comprehensive reviews focused on robotic navigation and driving control~\cite{kiran2021rl,prasuna2024deep,singh2025improvised}, deep reinforcement learning now underpins many end-to-end and modular autonomous driving systems.

Beyond foundational control, deep reinforcement learning has been applied to a wide array of increasingly complex autonomous driving tasks. For intersection navigation, studies like those by Ben Elallid et al.~\cite{benelallid2023intersection}, Shankar et al.~\cite{shankar2025deep}, and Spatharis and Blekas~\cite{Spatharis02012024} highlight the efficacy of deep reinforcement learning in minimizing delays and avoiding collisions in uncontrolled zones, while Jayawardana et al.~\cite{jayawardana2023} and Cederle et al.~\cite{cederle2024distributed} emphasize eco-driving and multi-agent reinforcement learning to distribute the decision-making load across connected vehicles. In the context of lane changing, Wang et al.~\cite{wang2024benchmarking} established benchmarks showing that reinforcement learning can effectively balance safety and traffic flow during merging maneuvers. Similarly, for car-following and freeway decision-making, Liu et al.~\cite{liu2024comparative}, Chen et al.~\cite{chen2023follownet}, and Marin et al.~\cite{Marin04122025} demonstrate that reinforcement learning frameworks can learn safe, human-like spacing and acceleration profiles, outperforming classical adaptive cruise control under variable traffic conditions. Robustness to environmental factors has also been explored; for example, studies by Tang et al.~\cite{tang2020} and Almalioglu et al.~\cite{almalioglu2022} address performance degradation in adverse weather, while Lee et al.~\cite{lee2024robust} and Ben Elallid et al.~\cite{BenElallid02112025} investigate control strategies designed to maintain stability across diverse daytime, rainy, and snowy scenarios. Other complex maneuvers, such as high-speed ramp merging~\cite{chen2023rampmerging} and overtaking~\cite{cui2023multiinput, li2020deepRL}, have similarly benefited from the ability to optimize trajectories dynamically, balancing the trade-off between progress and collision avoidance.

In complex continuous-control problems, reinforcement learning is frequently integrated into hierarchical or decoupled architectures. High-level planners generate reference trajectories or target velocities, which low-level agents then execute via smooth control commands~\cite{qiao2020hierarchical}. This separation of concerns aligns with established practices for improving system stability and interpretability. From a perception standpoint, vision-based setups—particularly those relying on single front-facing cameras—remain common and cost-effective. Studies on camera-only Bird’s-Eye-View perception~\cite{geiger2012kitti,busch2023improved,unger2023multicamera} and lightweight detection architectures~\cite{wu2022yolop,howard2017mobilenets,liu2016ssd} confirm that vision-only systems can achieve the scalability and representation quality required for downstream decision-making.

A significant portion of the foundational literature focuses on the specific task of lane following, which serves as a critical testbed for studying continuous control algorithms. Studies in this area have aimed to optimize different facets of control: Hua et al.~\cite{hua2022exploration} explored improved exploration strategies to enhance the stability of DDPG-based controllers, while Liu et al.~\cite{liu2024highspeed} proposed entropy-regularized approaches specifically tailored for high-speed lane tracking. Comparative analyses are also prevalent; Sabbir~\cite{sabbir2025comparative} contrasted value-based and policy-gradient methods to identify trade-offs in sample efficiency and final performance. Despite the widespread use of deep reinforcement learning, broad systematic comparisons between dominant algorithms—such as Proximal Policy Optimization (PPO)~\cite{schulman2017proximal}, Deep Deterministic Policy Gradient (DDPG)~\cite{lillicrap2015continuous}, Twin Delayed DDPG (TD3)~\cite{fujimoto2018addressing}, and Soft Actor-Critic (SAC)~\cite{haarnoja2019sac}—often yield varying results. While some studies evaluate these methods under restricted conditions like autonomous racing or sharp turns~\cite{evans2024racing,li2025sharpturns}, recent broader analyses~\cite{liu2024evaluation,xu2024lanechanging,park2025comparative} increasingly suggest that stability-enhanced and entropy-regularized actor-critic methods offer superior resilience and convergence properties.
"""

import re

# split text into sentences roughly
sentences = re.split(r'(?<=\.)\s+', text.strip().replace('\n\n', ' '))

ref_to_sentence = {}
for sentence in sentences:
    matches = re.findall(r'\\cite\{([^}]+)\}', sentence)
    if matches:
        keys = []
        for m in matches:
            keys.extend([k.strip() for k in m.split(',')])
        for k in keys:
            if k not in ref_to_sentence:
                ref_to_sentence[k] = []
            ref_to_sentence[k].append(sentence)

output = "# Exact Citations in Related Work\n\nThis document shows exactly how each of the 47 references is described in the newly rewritten 'Related Work' section of the article.\n\n"

for key in sorted(ref_to_sentence.keys()):
    output += f"### {key}\n"
    for s in ref_to_sentence[key]:
        output += f"> {s}\n\n"

with open('/home/ruben/Desktop/2020-phd-ruben-lucas/docs/assets/articles/2026_article_attempt/references_analysis.md', 'w') as f:
    f.write(output)
