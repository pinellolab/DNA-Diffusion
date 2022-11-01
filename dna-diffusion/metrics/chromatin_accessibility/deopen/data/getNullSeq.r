#==================================================================
# Title     : getNullSeq function, part of MotifRaptor2 repository.
# Author    : Manuel Tognon
# Date      : October 26, 2022
#==================================================================

if (requireNamespace("GenomicRanges", quietly = TRUE)&
    requireNamespace("rtracklayer", quietly = TRUE)&
    requireNamespace("BSgenome", quietly = TRUE)){
  
genNullSeqs = function(
  inputBedFN, 
  genomeVersion='hg19', 
  outputBedFN = 'negSet.bed', 
  outputPosFastaFN = 'posSet.fa',
  outputNegFastaFN = 'negSet.fa', 
  xfold = 1, 
  repeat_match_tol = 0.02, 
  GC_match_tol = 0.02, 
  length_match_tol = 0.02, 
  batchsize = 5000, 
  nMaxTrials = 20, 
  genome = NULL  # if genome is provided, genome version is ignored 
){
    
    #inputBedFile = '~/Downloads/ctcfpos.bed'
    #xfold = 1; 
    if(is.null(genome)){  
      if(toupper(genomeVersion)=='HG18'){
        if(requireNamespace("BSgenome.Hsapiens.UCSC.hg18.masked", quietly = TRUE)){
          genome <- BSgenome.Hsapiens.UCSC.hg18.masked::BSgenome.Hsapiens.UCSC.hg18.masked
        }
      } else{
        if(requireNamespace("BSgenome.Hsapiens.UCSC.hg19.masked", quietly = TRUE)){
          genome <- BSgenome.Hsapiens.UCSC.hg19.masked::BSgenome.Hsapiens.UCSC.hg19.masked
        }
      }
    }
    
    seqnams= GenomeInfoDb::seqnames(genome)
    chrlens = GenomeInfoDb::seqlengths(genome)
    chrpos = cumsum(as.numeric(chrlens))
    pmax = max(chrpos)
    chrpos = c(chrpos,1E12)
    chrpos0 = c(0, chrpos)
    ichrA = as.character(names(chrlens)); 
    
    
    getichrpos = function(ipos){
      j = order(ipos); 
      ipos = sort(ipos); 
      ci = 1;
      res = rep(NA, length(ipos))
      for(i in 1:length(ipos))
      {
        while(ipos[i]>chrpos[ci]){
          ci = ci+1; 
        }
        res[j[i]] = ci
      }
      return(res); 
    }
    
    generateRandomGenSeqs = function(seqlens){
      rpos = sample(pmax, length(seqlens), replace = TRUE)
      ichr1 = getichrpos(rpos)
      ichr2 = getichrpos(rpos+seqlens)
      
      jj = which(ichr1!=ichr2)
      while(length(jj)>0){
        rpos[jj] = sample(pmax, length(jj), replace = TRUE)
        ichr1 = getichrpos(rpos)
        ichr2 = getichrpos(rpos+seqlens)
        jj = which(ichr1!=ichr2)
      }
      chr = ichrA[ichr1]
      start = rpos - chrpos0[ichr1];
      names <- chr; 
      ranges <- IRanges::IRanges(start=start, width=seqlens)
      strand <- BiocGenerics::strand(sample(c("+", "-"), length(names), replace=TRUE))
      gr <- GenomicRanges::GRanges(seqnames=names, ranges=ranges, strand=strand)
    }
    
    inBed = rtracklayer::import.bed(inputBedFN)
    
    #check the BED file:
    inbed = GenomicRanges::as.data.frame(inBed)
    jj = which(is.na(match(as.character(inbed$seqnames),as.character(seqnams))))
    if (length(jj)>0){
      cat( paste('ERROR: Chromosome name not recognized for',length(jj), 'sequences.\n'))
      cat( unique(as.character(inbed$seqnames[jj])))
      return(NULL)
    }
    jj = which(inbed$end>GenomeInfoDb::seqlengths(genome)[as.character(inbed$seqnames)])
    if (length(jj)>0){
      cat( 'ERROR: Region outside chromosome. (Check the genome version) \n')
      print( inbed[jj,])
      return(NULL)
    }
    
    gcContent <- function(seqs)
    {
      alf <- Biostrings::alphabetFrequency(seqs, as.prob=TRUE)
      gc = rowSums(alf[,c("G", "C"), drop=FALSE])
    }
    # 
    # gc1=desGC[unmatched]
    # gc2=rndGC
    # len1=desLens[unmatched]
    # len2=width(rndBed)
    # rpt1=desRpt[unmatched]
    # rpt2=rndRpt
    # gc_th = GC_match_tol,
    # len_th = repeat_match_tol,
    # rpt_th = length_match_tol)
    
    matchSeqs = function(gc1, gc2, len1, len2, rpt1, rpt2,  gc_th=0.02, len_th=0.02, rpt_th=0.02){
      ##
      #  gc1 = rndGC; 
      #  gc2 = inGC; 
      #  len1 = seqlens; 
      #  len2 = seqlens;
      #  rpt1 = inRpt; 
      #  rpt2 = rndRpt; 
      #  gc1=desGC[unmatched];gc2= rndGC;len1= desLens[unmatched];len2= BiocGenerics::width(rndBed);rpt1= desRpt[unmatched];rpt2= rndRpt;
                      
      #  gc_th=0.02
      #  len_th=0.02
      
      
      # gc1, len1 are the desired 
      # gc2 and len2 are to be matched 
      len_th = len_th * len1; 
      
      i1 = order(gc1)
      i2 = order(gc2)
      
      gc1 = gc1[i1]
      gc2 = gc2[i2]
      len1 = len1[i1]
      len2 = len2[i2]
      rpt1 = rpt1[i1]
      rpt2 = rpt2[i2]
      
      gc2 = c(gc2, 1E10)
      
      len_th = len_th[i1]
      
      m2 = 1; 
      N = length(i1); 
      N2 = length(i2);
      mtc1 = rep(NA, N)
      mtc2 = rep(0, length(i2))
      for(i in 1:N){
        #if(i%%1000==0){cat(i,' ')}
        gc1i = gc1[i]; 
        len1i = len1[i]
        rpt1i = rpt1[i]
        len_thi = len_th[i]
        
        while(gc1i - gc2[m2]>gc_th) {
          m2 = m2+1; 
        }
        if(m2<=N2){
          m2b=m2;
          while(gc2[m2b]-gc1i<=gc_th){
            if ((mtc2[m2b]==0)&(abs(len1i-len2[m2b])<=len_thi)&(abs(rpt1i-rpt2[m2b])<=rpt_th)){
              mtc2[m2b]=i; 
              mtc1[i]=m2b;
              if(m2b==m2){m2 = m2+1;}
              break; 
            }
            m2b =m2b+1; 
          }
        }else{break;}
      }
      
      mtc1 = i2[mtc1]
      res = rep(NA, N)
      res[i1] = mtc1; 
      return(res)
    }
    
    #bed = rndBed
    repeatRat = function(bed){
      chrs = unique(GenomeInfoDb::seqnames(bed))
      rpts = rep(0, length(bed))
      
      for(ichr in as.character(chrs)){
        seq = genome[[ichr]]
        rpt = Biostrings::masks(seq)[['TRF']]
        
        jj = which(as.character(GenomeInfoDb::seqnames(bed))==ichr)
#       jj = which((GenomeInfoDb::seqnames(bed))==ichr)
        
        if (length(jj)>0){
          
          jbed = bed[jj]
          jrpts = rep(0, length(jj))
          
          olaps <- IRanges::findOverlaps(rpt, jbed@ranges)
          #isect <- pintersect(rpt[queryHits(olaps)], jbed@ranges[subjectHits(olaps)])
          
          qdf= GenomicRanges::as.data.frame(rpt)[S4Vectors::queryHits(olaps),]
          isect <- IRanges::pintersect(IRanges::IRanges(start=qdf$start,end=qdf$end), jbed@ranges[S4Vectors::subjectHits(olaps)])
          
          jres = S4Vectors::subjectHits(olaps)
          olap_width=BiocGenerics::width(isect)
          
          ## this could be done faster if duplicated(jres) is empty 
          for(i in 1:length(jres)){
            jrpts[jres[i]]=jrpts[jres[i]]+olap_width[i]
          } 
          rpts[jj] = jrpts;
        }
      }
      
      rpts = rpts/BiocGenerics::width(bed)
    }
    
    #check the BED file:
    inbed = GenomicRanges::as.data.frame(inBed)
    
    cat(' importing sequences for',inputBedFN, 'from', GenomeInfoDb::bsgenomeName(genome),'\n')
    #extract sequences
    inSeqs = Biostrings::getSeq(genome, inBed)
    seqlens = inbed$width
    inGC = gcContent(inSeqs)
    cat(' calculating repeat distributions\n')
    inRpt = repeatRat(inBed)
    
    nout = round(nrow(inbed)*xfold)
    #outbed=as.data.frame(matrix(ncol=ncol(inbed), nrow=nout))
    outbed=matrix(ncol=ncol(inbed), nrow=nout)
    outSeq = rep(inSeqs, length=nout); 
    
    colnames(outbed)=colnames(inbed)
    
    unmatched = 1:length(outSeq)
    
    desGC = rep(inGC, length=nout); #desired output GC
    desRpt = rep(inRpt, length=nout); #desired output repeat
    desLens = rep(seqlens, length=nout); #desired output lengths 
    
    for(iter in 1:nMaxTrials){
      if(length(unmatched)>0){
        cat(' Trial',iter,'out of',nMaxTrials,'\n')
        rndBed = generateRandomGenSeqs(rep(desLens[unmatched],length.out=batchsize))
        rndbed= GenomicRanges::as.data.frame(rndBed)
        cat(' importing sequences\n')
        rndSeqs = Biostrings::getSeq(genome, rndBed)
        rndGC = gcContent(rndSeqs)
        cat(' calculating repeat distributions\n')
        rndRpt = repeatRat(rndBed)
        cat(' matching sequences\n')
        
        mtc = matchSeqs(desGC[unmatched], rndGC, desLens[unmatched], BiocGenerics::width(rndBed), desRpt[unmatched], rndRpt,
                        gc_th = GC_match_tol,
                        len_th = length_match_tol,
                        rpt_th = repeat_match_tol)
        jj = which(!is.na(mtc))
        if(length(jj)>0){
          #outbed[unmatched[jj],]=rndbed[mtc[jj],];
          outbed[unmatched[jj],1:5]=as.matrix(rndbed[mtc[jj],]);
          outSeq[unmatched[jj],]=rndSeqs[mtc[jj],];
          unmatched = unmatched[-jj]
        }
        cat(nrow(outbed) - length(unmatched),'sequences found so far, ',length(unmatched), ' remaining.\n')
      }
    }  
    
    if(length(unmatched)>0){
      outbed = outbed[-unmatched,]
      outSeq = outSeq[-unmatched,]
    }

    outbed = gsub(' ','', outbed)
    write.table(as.matrix(outbed[,1:3]),quote = FALSE, sep='\t',row.names = FALSE, col.names = FALSE , file = outputBedFN)  
    if(requireNamespace("seqinr", quietly = TRUE)){
      outseqnams = paste(outbed[,1], outbed[,2], outbed[,3], 'neg', 1:nrow(outbed), sep='_')
      seqinr::write.fasta(sequences = sapply(as.character(outSeq), strsplit,''), names =outseqnams,file.out =   outputNegFastaFN); 
      inseqnams = paste(as.character(inbed[,1]), inbed[,2], inbed[,3], 'pos', 1:nrow(inbed), sep='_')
      seqinr::write.fasta(sequences = sapply(as.character(inSeqs), strsplit,''), names =inseqnams, file.out =   outputPosFastaFN); 
    }
    return(outputNegFastaFN)
    
  }
}

args = commandArgs(trailingOnly=TRUE)
library(BSgenome.Hsapiens.UCSC.hg38.masked)
if (length(args)==0) {
  stop("At least one argument must be supplied (Bed File)", call.=FALSE)
} else if (length(args)==1) {
  args[2]=10
} 

genNullSeqs(args[1], nMaxTrials=args[2], xfold=1, genome=BSgenome.Hsapiens.UCSC.hg38.masked);
ls
