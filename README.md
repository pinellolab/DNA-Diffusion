## Title
# Understanding the code of life: generative models of regulatory DNA sequences based on diffusion models.
<img src='https://raw.githubusercontent.com/pinellolab/DNA-Diffusion/f028558816fe5832097c270f424e3b3c3db48d8d/diff_first.gif'> </img>




## Abstract
__Provide a brief outline motivating the project. How would it positively impact biological research? What is the hypothesis behind it? No need to discuss datasets or models yet, we will do that later. Focus on the grand picture and \textit{why} the community should care about it.__
## Introduction and Prior Work
__Provide a short (preferably beginner friendly) introduction to the project and a brief outline of the literature most relevant to it. How does the project fit into this context?__

The goal of this project is to investigate the application and adaptation of recent diffusion models (see https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ for a nice intro and references) to genomics data. Diffusion models are powerful models that have been used for image generation (e.g. stable diffusion, DALL-E), music generation (recent version of the magenta project) with outstanding results. 
A particular model formulation called "guided" diffusion allows to bias the generative process toward a particular direction if during training a text or continuous/discrete labels are provided. This allows the creation of "AI artists" that, based on a text prompt, can create beautiful and complex images (a lot of examples here: https://www.reddit.com/r/StableDiffusion/).

Some groups have reported the possibility of generating synthetic DNA regulatory elements in a context-dependent system, for example, cell-specific enhancers. 
(https://elifesciences.org/articles/41279 , 
	https://www.biorxiv.org/content/10.1101/2022.07.26.501466v1)

The creation of DNA-regulatory elements is not limited to regulatory single isolated enhancers. Such idea can be expanded to create a whole regulatory locus with multiple promoters and their negative and positive regulators. Besides synthetic DNA creations, a diffusion model can help understand regulatory elements components and be a valuable tool for studying single nucleotide variations (https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1) and evolution.
(https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1502-5)

Based on these premises, we want to create a model that can generate cell type specific or context specific DNA-sequences with certain regulatory properties based on a simple prompt. For example: 

  - "Please generate a sequence that will activate a gene to its maximum expression level in cell type X"

  - "Please generate a sequence that will correspond to open (or closed) chromatin in cell type X"

  - "Please generate an activating sequence in cell type X that contains the transcription factor Gata1"


We believe this can accelerate our understanding of the intrinsic properties of DNA-regulatory sequence in normal development and different diseases.

## Proposed framework

For this work we propose to build a Bit Diffusion model based on the formulation proposed by Chen, Zhang and Hinton https://arxiv.org/abs/2208.04202. This model is a generic approach for generating discrete data with continuous diffusion models. An implementation of this approach already exists, and this is a potential code base to build upon:

https://github.com/lucidrains/bit-diffusion

## Tasks and potential roadmap:
  - Collecting genomic datasets 
  -Implementing the guided diffusion based on the code base
Thinking about the best encoding of biological information for the guided diffusion (e.g. cell type: K562, very strong activating sequence for chromatin, or cell type: GM12878, very open chromatin)
Plans for validation based on existing datasets or how to perform new biological experiments (we need to think about potential active learning strategies).




## Deliverables
What do we plan to provide the broader community with upon the completion of the project? Datasets? Models? APIs? Every deliverable should preferably have its own subsection with its associated potential impact, although it is not required.


  - __Dataset:__ compile and provide a complete database of cell-specific regulatory regions (DNAse assay) to allow scientists to train and generate different diffusion models based on the regulatory sequences.


  - __Models:__ Provide a model that can generate regulatory sequences given a specific cell type and genomic context.


  - __API:__ Provide an API to make it possible to manipulate DNA regulatory models and a visual playground to generate synthetic contextual sequences.


## Datasets
__If applicable, how large is the dataset that the project aims to produce? How difficult is producing such a dataset expected to be? What kind of resources are needed? What license will the dataset be licensed under? MIT is preferred but not required.__

- High-resolution maps of DHSs from 733 human biosamples encompassing 438 cell and tissue types and states, and integrated these to delineate and numerically index approximately 3.6 million DHSs within the human genome sequence (https://www.nature.com/articles/s41586-020-2559-3)


- DNA-sequences data corresponding to annotated regulatory sequences such as gene promoters or distal regulatory sequences such as enhancers annotated (based on chromatin marks or accessibility) for hundreds of cells by the NHGRI funded projects like ENCODE or Roadmap Epigenomics. 

- Data from MPRA assays that test the regulatory potential of hundred of DNA sequences in parallel (https://elifesciences.org/articles/69479.pdf , )

- MIAA assays that test the ability of open chromatin within a given cell type.




## Models
__If applicable, does the project aim to release more than one model? What would be the input modality? What about the output modality? How large are the models that the project aims to release? Are there other important differences between the models to be released? If the models are very different, consider writing a short subsection for each model type.__
## Input modality:
	A) Cell type + regulatory element ex: Brain tumor cell weak Enhancer
	B) Cell type + regulatory elements + TF combination (presence or absence) Ex: Prostate cell, enhancer , AR(present), TAFP2a (present) and ER (absent), 
	C) Cell type + TF combination + TF positions Ex: Blood Stem cell GATA2(presence) and ER(absent) + GATA1 (100-108)
	D) Sequencing having a GENETIC VARIANT -> low number diffusion steps = nucleotide importance prediction 	

### Output:
		DNA-sequence
__Model size:__
		The number of enhancers and biological sequences isn’t bigger than the number of available images on the Lion dataset. The dimensionality of our generated DNA outputs should not be longer than  4 bases [A,C,T,G] X ~1kb.  The final models should be bigger than ~2 GB.

__Models:__
		Different models can be created based on the total sequence length.

## APIs
__If applicable, what kind of API does the project aim to release? Are there any existing APIs that it could be integrated into? What kind of documentation could the project provide?__

## Paper
__Can the project be turned into a paper? What does the evaluation process for such a paper look like? What conferences are we targeting? Can we release a blog post as well as the paper?__

Yes, We intend to have a mix of our in silico generations and experimental validations to study our models' performance on classic regulatory systems ( ex: Sickle cell and Cancer).
Our group and collaborators present a substantial reputation in the academic community and different publications in high-impact journals, such as Nature and Cell.


## Resources Requirements
__What kinds of resources (e.g. GPU hours, RAM, storage) are needed to complete the project?__

COMPLETE


## Timeline
__What is a (rough) timeline for this project?__

## Broader Impact
__How is the project expected to positively impact biological research at large?__




## Reproducibility
__What steps are going to be taken to ensure the project's reproducibility? Will data processing scripts be released? What about training logs?__

We have several assays and technologies to test the synthetic sequences generated by these models at scale based on CRISPR genome editing or massively parallel reporter assays (MPRA).



## Failure Case
__If our findings are unsatisfactory, do we have an exit plan? Do we have deliverables along the way that we can still provide the community with?__

## Preliminary Findings
__If applicable, mention any preliminary findings (e.g. experiments you have run on your own or heard about) that support the project's importance.__


Using the Bit Diffusion model we were able to reconstruct 200 bp sequences that presented very similar motif composition to those trained sequences. The plan is to add the cell conditional variables to the model to check how different regulatory regions depend on the cell-specific  context



## Next Steps
__If the project is successfully completed, are there any obvious next steps?__ 
Expand the model lengh to generate complete regulatory regions (enhancers + Gene promoter pairs)  
Use our syntethic enhancers on in-vivo models and check how they can regulate the transcriptional dynamics in biological scenarios (Besides the MPRA arrays).  


## Known contributors
__Please list community members that you know are interested in contributing. It is best if a project proposal already has an associated team capable of going ahead with the project by themselves, but it is not necessary.__


__Luca Pinello,__ Associate Professor, Harvard Medical School,MGH Boston (lucapinello on Discord).   
__Wouter Meuleman,__ Investigator, Altius Institute for Biomedical Sciences & Affiliate Associate Professor, University of Washington, Seattle.  
__Lucas Ferreira,__ PostDoc, Harvard Medical School/MGH Boston.  
__Sameer Gabbita,__ High school intern, MGH, Student at Thomas Jefferson High School for Science & Technology.  
__Jiecong Lin,__ Postdoc, Harvard Medical School/MGH, Boston.  
__Zach Nussbaum,__ Machine Learning Engineer.  
