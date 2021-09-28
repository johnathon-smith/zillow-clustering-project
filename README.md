## Zillow Clustering Project
***

### Executive Summary - Conclusions and Next Steps
***

#### Goal
* Discover drivers of error in the Zestimate

#### Findings

* According to the important features property of my best model, the top drivers of logerror are structure value per square foot, size, and location.
* Unfortunately, since my model did not perform much better than baseline, I recommend further exploration using the features listed above as a starting point for clustering.

***

### Project Summary
***

#### Audience
* The Zillow data science team.

#### Deliverables
* A four minute verbal presentation.
* A Github Repository containing:
    - A clearly labeled final report jupyter notebook.
    - The .py files necessary to reproduce my work for anyone with their own env.py file.
* Finally, a README.md file documenting my project planning with instructions on how someone could clone and reproduce my project on their own machine. Goals for the project, a data dictionary, and key findings and takeaways should be included.

#### Context
* The Zillow data I'm using was acquired from the Codeup Database.

#### Data Dictionary (Relevant Columns Only)
| Target | Datatype | Definition |
|:-------|:---------|:------------|
| logerror | float | The Zestimate error |

| Feature | Datatype | Definition |
|:--------|:---------|:------------|
| bathroom_count | float | The number of bathrooms in the property (Includes values for half baths and other combinations) |
| quality_type | float | The quality rating of the property (1 - 10) |
| home_square_feet | float | The area of the property in square feet |
| latitude | float | The latitude of the property |
| longitude | float | The longitude of the property |
| room_count | float | The number of rooms the property has |
| county | str | The name of the county the property resides in |
| age | float | The age of the property in years |
| acres | float | The number of acres the property sits on |
| tax_rate | float | The tax rate of the property |
| structure_dollar_per_sqft | float | The value of the structure per square foot |
| land_dollar_per_sqft | float | The value of the land per square foot |
| bath_bed_ratio | float | The ratio of bathrooms to bedrooms of the property |

#### Initial Hypotheses

__Hypothesis 1__
* H_0: home_square_feet is not linearly correlated with logerror.  
* H_a: home_square_feet is linearly correlated with logerror.  
* alpha = 0.05

Outcome: Rejected the Null Hypothesis.

__Hypothesis 2__
* H_0: The average logerror for properties in Los Angeles county <= The average logerror for properties in Ventura county.
* H_a : The average logerror for properties in Los Angeles county > The average logerror for properties in Ventura county.
* alpha = 0.05

Outcome: Failed to Reject the Null Hypothesis.

__Hypothesis 3__
* H_0 : The average logerror for properties in Los Angeles county >= The average logerror for properties in Orange county.
* H_a : The average logerror for properties in Los Angeles county < The average logerror for properties in Orange county.
* alpha = 0.05

Outcome: Rejected the Null Hypothesis.

***

### My Process
***

##### Trello Board
 - https://trello.com/b/kbVX7e4p/clustering-project


##### Plan
- [x] Write a README.md file that details my process, my findings, and instructions on how to recreate my project.
- [x] Acquire the zillow data from the Codeup Database
- [x] Clean and prepare the zillow data:
    * Select only the useful columns
    * Remove or impute null values
    * Rename columns as necessary
    * Change data types as necessary
    * Remove entries that don't make sense or are illegal
    * Remove outliers
- [x] Plot individual variable distributions
- [x] Determine at least two initial hypotheses, run the statistical tests needed, evaluate the outcome, and record results.
- [x] Create at least three different cluster groups and perform statistical analysis to determine their usefulness.
- [x] Split the data sets into X and y groups 
- [x] Scale the X groups before use in the model.
- [x] Set baseline using logerror mean.
- [x] Create and evaluate model on train and validate sets.
- [x] Choose best model and evaluate it on test data set.
- [x] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire / Prepare
* Create and store functions needed to acquire and prepare the Zillow data from the Codeup Database in a wrangle.py file.
* Import the wrangle.py module and use it to acquire the data in the Final Report Notebook.
* Complete some initial data summarization (`.info()`, `.describe()`, ...).
* Plot distributions of individual variables.
* List key takeaways.

___

##### Plan -> Acquire / Prepare -> Explore
* Create visuals that will help discover new relationships between features and the target variable.
* Test initial hypotheses and record results.
* List key takeaways.
* Create three different cluster groups and visualize their relationships with the target variable.
* Use statistical testing to determine whether or not they will be useful in the model.

___

##### Plan -> Acquire / Prepare -> Explore -> Model / Evaluate
* Create dummy variables for the categorical features.
* Split data into X and y groups.
* Scale the X groups.
* Set a baseline using logerror mean.
* Create and evaluate at least four models on the train and validate data sets.
* Choose best model and evaluate it on the test data set.
* Document conclusions and next steps.

***

### Reproduce My Project

***

- [x] Read this README.md
- [ ] Download the final report Jupyter notebook.
- [ ] Download the wrangle.py, explore.py, and model.py modules into your working directory.
- [ ] Add your own env.py file to your directory. (user, password, host)
- [ ] Run the final report Jupyter notebook.