## Title
# Understanding the code of life: generative models of regulatory DNA sequences based on diffusion models.
<img src='https://raw.githubusercontent.com/pinellolab/DNA-Diffusion/f028558816fe5832097c270f424e3b3c3db48d8d/diff_first.gif'> </img>




## Abstract
The Human Genome Project has laid bare the DNA sequence of the entire human genome, revealing the blueprint for tens of thousands of genes involved in a plethora of biological process and pathways.
In addition to this (coding) part of the human genome, DNA contains millions of non-coding elements involved in the regulation of said genes. 

Such regulatory elements control the expression levels of genes, in a way that is, at least in part, encoded in their primary genomic sequence. 
Many human diseases and disorders are the result of genes being misregulated.
As such, being able to control the behavior of such elements, and thus their effect on gene expression, offers the tantalizing opportunity of correcting disease-related misregulation.

Although such cellular programming should in principle be possible through changing the sequence of regulatory elements, the rules for doing so are largely unknown.
A number of experimental efforts have been guided by preconceived notions and assumptions about what constitutes a regulatory element, essentialy resulting in a "trial and error" approach.

Here, we instead propose to use a large-scale data-driven approach to learn and apply the rules underlying regulatory element sequences, applying the latest generative modelling techniques.


## Introduction and Prior Work
The goal of this project is to investigate the application and adaptation of recent diffusion models (see https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ for a nice intro and references) to genomics data. Diffusion models are powerful models that have been used for image generation (e.g. stable diffusion, DALL-E), music generation (recent version of the magenta project) with outstanding results. 
A particular model formulation called "guided" diffusion allows to bias the generative process toward a particular direction if during training a text or continuous/discrete labels are provided. This allows the creation of "AI artists" that, based on a text prompt, can create beautiful and complex images (a lot of examples here: https://www.reddit.com/r/StableDiffusion/).

Some groups have reported the possibility of generating synthetic DNA regulatory elements in a context-dependent system, for example, cell-specific enhancers. 
(https://elifesciences.org/articles/41279 , 
	https://www.biorxiv.org/content/10.1101/2022.07.26.501466v1)


### Step 1: generative model

We propose to develop models that can generate cell type specific or context specific DNA-sequences with certain regulatory properties based on an input text prompt.
For example: 

  - "Please generate a sequence that will activate a gene to its maximum expression level in cell type X"

  - "Please generate a sequence that will correspond to open (or closed) chromatin in cell type X"

  - "Please generate an activating sequence in cell type X that contains the transcription factor Gata1"

  - "Please generate a sequence that activates a gene in liver and heart, but not in brain"


### Step 2: extensions and improvements

Beyond individual regulatory elements, so called "Locus Control Regions" are known to harbour multiple regulatory elements in specific configurations, working in concert to result in more complex regulatory rulesets. Having parallels with "collaging" approaches, in which multiple stable diffusion steps are combined into one final (graphical) output, we want to apply this notion to DNA sequences with the goal of designing larger regulatory loci. This is a particularly exciting and, to our knowledge, hitherto unexplored direction.

Besides synthetic DNA creations, a diffusion model can help understand and interpret regulatory sequence element components and for instance be a valuable tool for studying single nucleotide variations (https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1) and evolution.
(https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1502-5)


Taken together, we believe our work can accelerate our understanding of the intrinsic properties of DNA-regulatory sequence in normal development and different diseases.

## Proposed framework

For this work we propose to build a Bit Diffusion model based on the formulation proposed by Chen, Zhang and Hinton https://arxiv.org/abs/2208.04202. This model is a generic approach for generating discrete data with continuous diffusion models. An implementation of this approach already exists, and this is a potential code base to build upon:

https://github.com/lucidrains/bit-diffusion

## Tasks and potential roadmap:
  - Collecting genomic datasets 
  -Implementing the guided diffusion based on the code base
Thinking about the best encoding of biological information for the guided diffusion (e.g. cell type: K562, very strong activating sequence for chromatin, or cell type: GM12878, very open chromatin)
Plans for validation based on existing datasets or how to perform new biological experiments (we need to think about potential active learning strategies).


## Deliverables

  - __Dataset:__ compile and provide a complete database of cell-specific regulatory regions (DNAse assay) to allow scientists to train and generate different diffusion models based on the regulatory sequences.


  - __Models:__ Provide a model that can generate regulatory sequences given a specific cell type and genomic context.


  - __API:__ Provide an API to make it possible to manipulate DNA regulatory models and a visual playground to generate synthetic contextual sequences.


## Datasets

### DHS Index:
Chromatin (DNA + associated proteins) that is actively used for the regulation of genes (i.e. "regulatory elements") is typically accessible to DNA-binding proteins such as transcription factors ([review](https://www.nature.com/articles/s41576-018-0089-8), [relevant paper](https://www.nature.com/articles/nature11232)).
Through the use of a technique called [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), we've measured which parts of the genome are accessible across 733 human biosamples encompassing 438 cell and tissue types and states, resulting in more than 3.5 million DNase Hypersensitive Sites (DHSs).
Using Non-Negative Matrix Factorization, we've summarized these data into 16 _components_, each corresponding to a different cellular context (e.g. 'cardiac', 'neural', 'lymphoid').
These full datasets and annotations are available [here](https://www.meuleman.org/research/dhsindex/) and the paper is available [here](https://www.nature.com/articles/s41586-020-2559-3).

For the efforts described in this proposal, and as part of an earlier [ongoing project](https://www.meuleman.org/research/synthseqs/) in the research group of Wouter Meuleman, we've put together smaller subsets of these data that can be used to train models to generate synthetic sequences for each NMF component.
Specifically, we have the following datasets available:
- __training set:__ 160k sequences, 10k per NMF component ([.csv.gz](https://www.dropbox.com/s/db6up7c0d4jwdp4/train_all_classifier_WM20220916.csv.gz?dl=0)]
- __validation set:__ 16k sequences, 1k per NMF component ([.csv.gz](https://www.dropbox.com/s/fo98xtrkt0p26oo/validation_all_classifier.csv.gz?dl=0)]
- __test set:__ 16k sequences, 1k per NMF component ([.csv.gz](https://www.dropbox.com/s/nuglvfpy2za6gy3/test_all_classifier.csv.gz?dl=0)]

Each of these contains the genomic locations (human genome assembly hg38, first 3 columns) of accessible genome elements, their majority NMF component (column: 'component') as well as their nucleotide sequence (column: 'raw_sequence').


### Other potential datasets:

- DNA-sequences data corresponding to annotated regulatory sequences such as gene promoters or distal regulatory sequences such as enhancers annotated (based on chromatin marks or accessibility) for hundreds of cells by the NHGRI funded projects like ENCODE or Roadmap Epigenomics. 

- Data from MPRA assays that test the regulatory potential of hundred of DNA sequences in parallel (https://elifesciences.org/articles/69479.pdf , )

- MIAA assays that test the ability of open chromatin within a given cell type.

## Models

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
TBD depending on interest

## Paper
__Can the project be turned into a paper? What does the evaluation process for such a paper look like? What conferences are we targeting? Can we release a blog post as well as the paper?__

Yes, We intend to have a mix of our in silico generations and experimental validations to study our models' performance on classic regulatory systems ( ex: Sickle cell and Cancer).
Our group and collaborators present a substantial reputation in the academic community and different publications in high-impact journals, such as Nature and Cell.


## Resources Requirements
__What kinds of resources (e.g. GPU hours, RAM, storage) are needed to complete the project?__

Our initial model can be trained with small datasets (~1k  sequences) in about 3 hours ( ~500 epochs) on a colab PRO (24GB ram ) single GPU Tesla K80. Based on this we expect that to train this or similar models on the large dataset mentioned above  ( ~3 million sequences (4x200) we will need several high-performant GPUs for about 3 months. ( Optimization suggestions are welcome!)

## Timeline
__What is a (rough) timeline for this project?__

6 months to 1 year.

## Broader Impact
__How is the project expected to positively impact biological research at large?__

We believe this project will help to better understand genomic regulatory sequences: their composition and the potential regulators acting on them in different biological contexts and with the potential to create therapeutics based on this knowledge. 


## Reproducibility
We will use best practices to make sure our code is reproducible and with versioning. We will release data processing scripts and conda environments/docker to make sure other researchers can easily run it.

We have several assays and technologies to test the synthetic sequences generated by these models at scale based on CRISPR genome editing or massively parallel reporter assays (MPRA).


## Failure Case
Regardless of the performance of the final models, we believe it is important to test diffusion models on novel domains and other groups can build on top of our investigations.

## Preliminary Findings
Using the Bit Diffusion model we were able to reconstruct 200 bp sequences that presented very similar motif composition to those trained sequences. The plan is to add the cell conditional variables to the model to check how different regulatory regions depend on the cell-specific context.


## Next Steps
Expand the model lengh to generate complete regulatory regions (enhancers + Gene promoter pairs)  
Use our syntethic enhancers on in-vivo models and check how they can regulate the transcriptional dynamics in biological scenarios (Besides the MPRA arrays).  

If this project sounds exciting to you, **please join us**!


## Known contributors 
__Luca Pinello,__ Associate Professor, Harvard Medical School,MGH Boston (lucapinello on Discord).   
__Wouter Meuleman,__ Investigator, Altius Institute for Biomedical Sciences & Affiliate Associate Professor, University of Washington, Seattle.  
__Lucas Ferreira,__ PostDoc, Harvard Medical School/MGH Boston.  
__Sameer Gabbita,__ High school intern, MGH, Student at Thomas Jefferson High School for Science & Technology.  
__Jiecong Lin,__ Postdoc, Harvard Medical School/MGH, Boston.  
__Zach Nussbaum,__ Machine Learning Engineer.


