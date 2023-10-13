# Dataset for Recommending Scientific Datasets

Create a dataset for recommendation of scientific datasets using the [DataFinder](https://aclanthology.org/2023.acl-long.573.pdf) data.  

Augment the DataFinder dataset by curating the details of the datasets from PwC.  

Original DataFinder task: Recommend relevant scientific dataset for a research question related query.  
We reformulate the above task into a ***MCQ question-answering format***.  

The dataset is available on huggingface.

The task is available in two types of MCQs:  
I. MCQs with four options and a single correct answer [[link](https://huggingface.co/datasets/shrutisingh/dataset_recommendation_mcq_sc)]  
II. MCQs with variable number of options and multiple correct answers [[link](https://huggingface.co/datasets/shrutisingh/dataset_recommendation_mcq_mc)]

Please note that we ensure that there is no intersection in the queries in both formats. This ensures that there is no data leakage and both datasets are independent. Any model trained on the dataset for the first task should not lead to seeing the answers for queries in the second dataset.


It would be interesting to find out if a model trained on the first task can learn to select multiple answers from diverse number of choices correctly in comparison to a closed set of answers.


An example of both datasets is provided below:  
`I. MCQ with four options and a single correct answer`
<pre>
<b>Query:</b> A neural network model to solve structure-from-motion.
<b>Keyphrases: </b> structure-from-motion (sfm) video
<b>Options: </b>
A. AIDS  B. WikiReading  C. KITTI  D. AIDS Antiviral Screen
<b>Dataset details:</b>
AIDS Antiviral Screen Dataset is ...
WikiReading is a large scale NLU task and publicly available dataset with 18M instances ...
:

<b>Correct answer:</b> KITTI
</pre>

`II. MCQs with variable number of options and multiple correct answers`

<pre>
<b>Query:</b> I want to implement a real-time action detection system.  
<b>Keyphrases:</b> action detection video
<b>Options:</b>
A. UCF101  B. ESAD  C. G3D  D. COCO  E. SoccerDB ... (variable number of options in each instance)
<b>Dataset details:</b>
UCF101 dataset is an extension of UCF50 and consists of 13,320 video clips, which are classified into 101 categories. These 101 categories ....
ESAD is a large scale dataset ...
:

<b>Correct answer:</b> UCF101, COCO
</pre>


## Setup the repository:

```conda create -n datarecommender python=3.11```  
```pip install -r requirements.txt```


## Loading the dataset
```
from datasets import load_dataset
mcq_sc = load_dataset('shrutisingh/dataset_recommendation_mcq_sc')
mcq_mc = load_dataset('shrutisingh/dataset_recommendation_mcq_mc')
```