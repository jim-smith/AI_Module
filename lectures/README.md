# README

## Introduction
### Week 1:
Week 1: Intro to module and types of problems
- Learning outcomes
 - Familiarity with module contents, learning style, assessments
 - Understanding of basic input-model-out and hence optimisation/modelling / prediction
and the relationship to abduction/induction/deduction
 - Recognising the key differences between making a model and using a model
 - Recognising the difference between supervised learning and expert system elicitation
 - Clone a notebook from github and open in Jupyter
 - Run their first notebook to create a very simple chatbot
- Lecture materials
 - 15 minutes module-intro.ipynb
   - Activity: Explore the module website in order to find out the answer to some key questions: Nathan’s email, c/w hand-in dates, ‘what are the three topics covered’ book’
 - 15 minutes types_of_problems.ipynb
   - Activity: self-assessment test 1: identifying types of problem in angry birds
 - 15 minutes Symbolic-vs-Computational_AI.ipynb  
   - Activity: Add one example each of optimisation and modelling problem  to a wiki and comment on two other comments

- Tutorial recap  questions using mentimeter covering: Type of problems, types of logic, "machine learning deals with equations,  expert systems deal with assertions"
     
- Tutorial activity
 - Introduction to github / Microsoft azure/ google colab] and notebooks
 - First notebook:  tutorials/workbook1_a_trivial_chatbot.ipynb 
 - Using a simple chatbot to experiment with different capitalisation and punctuation to understand what preprocessing happens. Built in Multiple choice questions to check understanding
- Reading and Q’s for next week: https://developer.ibm.com/articles/cc-beginner-guide-machine-learning-ai-cognitive/


## Topic 1: Knowledge Representation
### Week 2: Representing knowledge: Components of knowledge
Learning Outcomes: 
 - Ability to represent human knowledge in the form of sets of rules and facts
 - Ability to recognise that knowledge exists at different levels: 
 - (meta knowledge, domain-specific vs generic, ontologies)
 - Specify and implement a simple rule-based expert system with forward chaining in AIML
Lecture Materials:
 - 10 minutes: Knowledge_Representation_Rules_and_Facts.ipynb
 - Activity Questions: 
  - fact or rule: for a flat triange ABC, the angle at B is 34 degrees; The angle at C is 56 degrees; The sum of the angles in a triangle is 180 degrees 
  - T/F: if my observations are correct and I apply a rule, then my outcome is correct.
  - T/F: if my axioms are valid and my observation is correct then any inferences I make are valid
  - Multiple Answer:/A Why do we say _“my digital scales tell me my apple has mass 56g”_ is an observation?
    – time variant (dries out),
    - human error in measurement (perhaps didn't zero the scales),
    - even simple machines like scales have limited precision and repeatability,  
    - misleading (I may be trying to con you)
 - 10 minutes Knowledge_Representation_Meta Knowledge.ipynb : properties of relationships, ontologies (e.g. schema.org) , long-term vs short term, domain-specific vs generic
   - Qs:
 - 15 minutes: AIML_an_example_KRL.ipynb : Expert systems basic flow, AIML atomic categories encode rules and facts, Forward chaining,  AIML srai allows the implies connective 
 - Short video where I point out all the meta knowledge we implicitly use to solve it?
 - Activity: BB quiz to complete the puzzle

Tutorial recap  questions using mentimeter:
 - different types of knowledge, rules vs facts, metaknowledge
 - forward chaining, '<category>', '<pattern>', '<template>', '<srai>'
Tutorial Activity: 
 - Identify some definitions of key concepts from the lectures
 - ../tutorials/workbook2-authoring-your-first-chatbot.ipynb
 - Notebook creating a chatbot to provide those definitons when asked.
 - using srai  to do forward chaining (Hi/Hello=>greeting,  but not wildcards yet)
Self-study: add more categories to revision bot

Stretch activity: identify facts about languages- different ways of asking questions

Reading : https://www.gamasutra.com/view/feature/3761/beyond_aiml_chatbots_102.php?print=1 , 
https://plato.stanford.edu/entries/science-theory-observation/
    
### Week 3: Representing knowledge: variables, generalisation
Learning outcomes:  
- recognition that deduction can be a multi-step process
- familiarity with the idea of variables that get bound to specific objects and quantifiers as ways of expressing knowledge  about groups of objects
- Illustrate that how you represent knowledge affects what you can represent
- Create more complex aiml model using wildcards, think etc as per AIMLv1
- Multistep dialogue for classifying fruit
Lecture materials: 
- 10 minutes: Knowledge_Representation_Syntax_and_Semantics.ipynb : Types of reasoning, closed-world vs open world, Syntax vs semantics , idea of a Knowledge Representation Langiage as a design choice.
- 10 minutes: Knowledge_Representation_Variables_and_Quantifiers.ipynb : Connectives, variables, quantifiers, Conflict resolution, FOL vs propositional logic, decideability
- 10 minutes:  AIML_variables_and_branching.ipynb
- 10 minutes:  AIML_context.ipynb : `<that>` and `<topic>`, AIML rule priorities

Tutorial recap  questions using mentimeter, covering: examples of short-term and generic vs domain specific knowledge, reminder about preprocessing (  ‘wasn’t -> ‘was not’ was dealt with by pre-processor but knowledge about module tutors could be in aiml file), questions on angry birds, `<star>`, `<that>`,`<topic>`, `<think>`, `<condition>`
    
Tutorial activities: tutorials/workbook3-using-variables-and-context-in-a-chatbot.ipynb
- identify different ways of asking question
- using wildcards/* to add them as a separate aiml file
-  jokebot for <that>
- Introduction to first coursework
 - Download Coursework/AIML_Coursework_marker.ipynb or .py, coursework-questions-and-responses.txt, 1cat.aiml
 - Make sure students can get them working


Self-study: update your revision bot with this week;s core concepts

### Week 4: Applications of knowledge based systems
Learning Outcomes
- understanding of the wider application of knowledge based systems,
- ability to describe the strengths and weaknesses of knowledge-based approaches and recognise when it would be appropriate to use them

Lectures: Modern-expert-Systems.ipynb
- 10 minutes: use of expert systems for goal-driven search (planning/optimisation)
 - Forward and backward chaining : example using trivial knowledge base(john-mary-thief-cheese)
 - Mycin as a classic example
 - Activity: question in middle: does john steal cheese? 
- 20 minutes: other semantic web, ontologies e.g. wordNet, but also Wikipedia, schema.org, OpenStreetMap etc.    ‘hidden’ expert systems e.g. in games IDEs (Unity Rule Tile) 
- 15 minutes: criticisms of expert systems and extensions (e.g. aIML v2), issues around provenance
- Activity:

Tutorial recap  questions using mentimeter covering: What can you induce from forward and backward chaining, Current ethical issues around provenance, fake news etc., Closed world: no “NOT” in AIML but default class
    
Tutorial activities:
- Activity 1:Continue notebook to work on first part of coursework
- Activity: Notebook to make calls out from aiml to google  etc. and to let them set predicates


Self-study: 
- update your revision bot with this week;s core concepts
- submit coursework
    
    
## Topic 2: Machine Learning
    
### Week 5
Learning Outcomes
- Identify and illustrate the legal and ethical issues around the use of machine learning
- Identify and illustrate the difference between unsupervised, reinforcement and supervised learning
- Formulate and apply simple unsupervised learning algorithms and illustrate their use.
    
Lectures: Machine_Learning_Introduction_and_Types.ipynb
- 15 minutes: types of ML
- Activity:
- 15 minutes: Unsupervised leaning with  k-Means example for Iris data
- Activity:
    TODO kmean coding example
- 15 minutes: Reinforcement Learning (popular video) simple code?

Tutorial recap  questions using mentimeter:
- Types of problem and algorithm from games, recommender systems
    
Tutorial activity
- Notebook using plotly libraries to create some visualisations for versions of the iris data then apply k-means from scikit-learn, and use this to illustrate considerations for distance-based models like the need for normalisation, effect of noisy/irrelevant features
 - Use the tutorial plan from 19/20 week 6 but might need a lot of help to make these interactive.  **Avoid seaborn?** just use pandas scatter_matrix() plots?
 - load iris data , set labels - blank for all of them
 - pplot scatter matrix: I'll have ot provide pyplot code to do that
 - run KMeans in 4D then repeat scatter_matrix with colouring   look into cluster matrix space
 - run KMEans in 2D and do colouring ( )
 - now normalise the data and rerun- do you get better clusters?
    

- Stretch activity: apply algorithms to the artificial fruit  dataset from week1  

Self-study: 
- Update your revision bot with this week;s core concepts
- Really nice explanation of Q-learning here: http://mnemstudio.org/path-finding-q-learning-tutorial.htm
    

### Week 6: Supervised Machine Learning 
Learning Outcomes
- Identify and illustrate the legal and ethical issues around the use of supervised machine learning
- Identify formulate and apply the basic processes of supervised machine learning
- Understand the role of data in estimating accuracy 
Lectures: Supervised_Machine_Learning.ipynb
- 15 minutes: basic model building process: train and test (Validation and model selection are in L2)
- Activity:
- 15 minutes:  types of model: instance-based (kNN) vs explicit (decision trees,rules) 
- Activity:
- 15 minutes: Exaxmple-  greedy rule induction as compared to expert system
- Activity:

Tutorial recap  questions using mentimeter:
- Train, test data. Examples of different results from 1-NN and 3-NN. Identify the first rule created …(old BB question) Classifier design and performance for multiclass problems
- discussion of ethical issues    
Tutorial activity
- Notebook applying sklearn k-NN to iris data,. Going through the process of train-test split,   data preprocessing/normalisation.
- Stretch activity: repeat for  MNIST 

Self-study: update your revision bot with this week;s core concepts



### Week 7: Artificial Neural Networks 1:Perceptrons 
Learning Outcomes
- Principles of biological and artificial neural networks
- Understand the perceptron update rule and the principles of the operations of MLPs
Lectures: ANN1_Perceptrons.ipynb
- Perceptrons as a way of learning  rules that aren’t axis-parallel
- Activity:
- Second half of perceptrons
Tutorial recap  questions using mentimeter: axis-parallel vs. oblique rules
Tutorial activity 
- (Nathan's) Notebook applying perceptron to logical functions and then fake data
- Description of coursework and discussion about what will be needed but in a way that makes it clear they can’t start yet e.g. framework(.py files) not released **could split into tasks e.g. design, tesat plan, spes**
Self-study: update your revision bot with this week's core concepts

### Week 8: Artificial Neural Networks 2: MultiLayer Perceptrons 
Learning Outcomes
- Principles of biological and artificial neural networks
- Understand the perceptron update rule and the principles of the operations of MLPs
Lectures:ANN2_MLP.ipynb
- 20 minutes: Multi Layer Perceptrons – architecture and principle of stochastic gradient descent examples: iris, MNIST
- Activity:
- 10 minutes: classification vs regression example applications: MNIST, lkeert scale??, 
- Activity:
- 15 minutes: problem formulation for multi-class problems one-vs-all vs softmax,
- Activity:
Tutorial recap  questions using mentimeter:
Tutorial activity
- Notebook applying MLP to iris data and then MNIST
- When completed auto-release notebook and framework for implementing greedy rule induction for iris data and coursework
- Stretch activity: notebook using and visualising  MLP for MNIST – tune topology

Self-study: 
- update your revision bot with this week;s core concepts
- deep learning resources from 2019-20 thatr do mnist with conv nets??
    
## Topic 3: Search
Week 9: Problem Solving as search
Learning Outcomes:
- Recognise and characterise Problem solving as search
- Formulate problems via representations of candidate solutions allowing the use of standard algorithms, search on a graph 
- Able to characterise properties of search algorithms
Lectures: recap of input-model-output then
- Problem solving as search through space of  representations of candidate solutions
- The process: constructive  vs perturbative (holistic) approaches (Illustrate via (reference to greedy rule induction and greedy TSP) but also forward chaining “when you’ve eliminated the probable but impossible …”)
- Activity:
- Properties of search algorithms (complete, efficient, optimal) wrt design tradeoffs (use scaleability of rulesets and TSP as an example)
- Activity:

Tutorial recap  questions using mentimeter:
- Properties of algorithms, Tradeoffs for different applications, contents of search space for different type of problem, recognising constructive and perturbative algorithms in other settings (e.g. satnav - progreessive refinement from initial waypoints - video would be nice)
Tutorial activity
- Hand-coding search algorithm for pacman?? Ideally with a gif unrolling a maze into a graph
- Stretch Activity: exhaustive search for code cracking
- Or …Greedy Rule Induction coursework

Self-study: update your revision bot with this week's core concepts

### Week 10: search algorithms for decision problems
Learning Outcomes
- Understand how different single-member search algorithms fit into common ‘generate-and-test’ framework
- Blind search for decision problems, formulation and algorithms 
Lectures: 
- 20minutes: Decision problems and appropriate algorithms– depth-first, breadth-first
- 20 minutes: Constraints,  move operators and neighbourhood structures
- 10 minutes: Implications wrt scaleability etc (password strength)
- Activity:

Tutorial recap  questions using mentimeter: Recognising algorithms from descriptions, identifying appropriate algorithms

Tutorial activity
- Notebook with framework, implementing algorithms to crack codes, then reapply for NQueens (one perturbative, one constructive, emphasize similarities)

Self-study: update your revision bot with this week;s core concepts

### Week 11: search algorithm guided by a ‘cost’ function  
Learning Outcomes
- Understand how different single-member search algorithms fit into common ‘generate-and-test’ framework
- Informed search algorithms
Lectures: recap then
- 15 minutes Quality functions: design choices,  landscape metaphor, 
- Activity:
- 15 minutes hill-climbing, A* but also mention best-first. Mention coursework and that SGD is a hill-climber
- Activity:
- 20 minutes: Recognising the problems of single member search: local optima, scaleability
Tutorial recap  questions using mentimeter:

Tutorial activity
- Implementing A* for a pathfinding application (e.g. a NPC controller in a game)
Self-study: update your revision bot with this week's core concepts

