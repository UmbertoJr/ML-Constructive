The ML-Constructive Heuristic
=========
The general idea
----------
ML-Constructive (ML-C) is the first constructive heuristics 
that scales efficiently using information extracted from historical TSP optimal tours.
ML-C exploits machine learning (ML) to learn common patterns from known optimal 
solutions provided by a perfect Oracle (the Concorde optimal TSP solver).
Then, such learnt ability is employed by the ML-C heuristic 
to construct TSP tours in two phases.

The first phase uses ML to select the most promising edges to insert in an initial fragmented 
solution (Figure 1).
The ML choose edges from the selection of promising edges.
Such selection is distinguished by the simple fact of collecting edges that are connection 
between vertices that are at most second closest to each other.


<p align="center">
	<img src="figures\firstphase.png" alt="example plot"/>
    <em>Figure 1</em>
</p>

Such partial solution enables reducing the construction time for solving, 
meanwhile smartly selecting the most promising edges.
The ML system chooses which edges to include in the solution and which not by employing 
image processing.
Three channels are utilized to represent a local view of the problem.
The first one shows the vertices in the local view,
the second one shows the new edge to be inserted, 
while the third channel shows the existing edges available in the current partial solution.
For more details we refer to Mele *et al.* [1]. 

<p align="center">
	<img src="figures\channels.png" alt="example plot"/>
    <em>Figure 2</em>
</p>


The second (and last) phase of ML-C uses the Clarke-Wright heuristic 
to complete the TSP tour (Figure 2). 
Several approaches are possible for completing the tour.
In Vitali *et al.* [2], for example,
a third phase employing local search techniques on the edges inserted 
during the second phase is suggested.

<p align="center">
	<img src="figures\secondphase.png" alt="example plot"/>
    <em>Figure 3</em>
</p>

The approach shows good result on TSPLIB testing instances up to 1748 vertices.
ML-C is effectively able to combine classical heuristic methods with machine learning 
by learning from small instances and scaling the knowledge to real case scenario.

For the curious readers, we suggest to look through our survey on ML approaches for the TSP [3], 
and an unusual example on how to combine combinatorial optimization concepts 
with the reward function of reinforcement learning setups for the TSP [4]. 





Dependecies
-----
Python>=3.8.8

Pytorch>=1.7

Cython>=0.29.23

git+git://github.com/jvkersch/pyconcorde.git

H5py>=2.10.0

Numpy>=1.20.2

Pandas>=1.2.2

Scipy>=1.6.2

Matplotlib>=3.3.4

Tqdm>=4.59.0

and others, please refer to requirements.txt for additional information.

How to install?
------

```shell
git clone https://github.com/UmbertoJr/ML-Constructive.git
cd ML-greedy/version1/
```


Dataset creation
------
The data creation it takes about 3 days.
After installing [Pyconcorde](https://github.com/jvkersch/pyconcorde) 
and the other dependencies correctly.
The folders and files containing the data for training 
and evaluation will be created with the following command. 
The Test instances are already located in the folder "version1/data/test".

```shell
python cli.py --operation create_instances
```

Statistical test
----
To reproduce the statistical results 
shown in Mele *et al.* [1], the following command is employed.
```shell
python cli.py --operation statistical_study
```

Train on random instances
-----
```shell
python cli.py --operation train
```

Test on TSPLIB instances
----
```shell
python cli.py --operation solve_TSPLIB
```

References
-----
<a id="1">[1]</a>
Mele, U. J., Gambardella, L. M., and Montemanni, R. (2021).
A New Constructive Heuristic driven by Machine Learning for the traveling Salesman Problem.
(submitted for publication).

<a id="1">[2]</a>
Vitali, T., Mele, U. J., Gambardella, L. M., and Montemanni, R. (2021).
Machine Learning Constructives and Local Searches for the Travelling Salesman Problem. 
ArXiv preprint ArXiv:2108.00938

<a id="1">[3]</a>
Mele, U. J., Gambardella, L. M., and Montemanni, R. (2021).
Machine learning approaches for the traveling salesman problem: A survey.
In Proceedings of the 8th International Conference on Industrial Engineering and Applications (ICIEA 2021).
Association for Computing Machinery (in press).

<a id="1">[4]</a>
Mele, U. J., Chou, X., Gambardella, L. M., and Montemanni, R. (2021).
Reinforcement Learning and additional rewards for the traveling salesman problem.
In Proceedings of the 8th International Conference on Industrial Engineering and Applications (ICIEA 2021).
Association for Computing Machinery (in press).


Cite
----
```buildoutcfg
@article{mele2021mlconstructive,
    title   = {A New Constructive Heuristic driven by Machine Learning for the Traveling Salesman Problem},
    author  = {Mele, Umberto Junior and Gambardella, Luca Maria and Montemanni, Roberto},
    journal = {Submitted for publication},
    year    = {2021}
}
```