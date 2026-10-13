# Automated extraction of evidence level from guidelines

Gist: automatically extract evidence level from medical recommendations from guideline documents using a LLM

Use case: 
- given the pdf of a guideline, extract each recommendation, along with the strength of recommendation and level of evidence
- give a numerical summary of the average strength of recommendation and average level of evidence backing the guideline

Strengths of recommendations are: 
A- strongly support a recommendation for use
B- moderately support a recommendation for use
C- marginally support a recommendation for use 
D- recommendation against

Level of evidence grading: 
1- Evidence from at least one properly designed randomised, controlled trial; with the primary objective of the study aligned with the recommendation being made.
2- Evidence from at least one well-designed clinical trial, without randomisation; from cohort or case-controlled analytic studies (preferably from more than one centre); from multiple time series; or from dramatic results of uncontrolled experiments.
3- Evidence from opinions of respected authorities, based on clinical experience. descriptive case studies, or reports of expert committees.

## Data
example data extracted by experts, that can be used for finetuning and testing of models
in each file there is the link to the paper from which the pdf can be downloaded, along with all recommendations in the paper along with their level of evidence and strengh of recommendation

path to data: /mnt/data1/klug/datasets/evidence_extraction

## Example analysis
- https://jamanetwork.com/journals/jama/fullarticle/2728486

 
 
