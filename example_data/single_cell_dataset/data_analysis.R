load("./rdata/pca.RData") ## Computed in ./data_parsing_full_dataset.R
index <- sample(seq(1,405191,1),20000)
label <- tissues[index]
write.csv(label,file="single_cell_label.csv",row.names = F)
subset <- pca[index,]
label <- read.csv("single_cell_label.csv")

library(reticulate)
use_python("C:/Users/17426/Anaconda3/python.exe",TRUE) ##Change path to python installation with umap-learn installed if necessary
umap_module=import("umap",convert=TRUE)

## UMAP
umap=umap_module$UMAP(n_neighbors=30L,min_dist=0.2,metric="euclidean",verbose=TRUE)$fit_transform(subset)
umap <- data.frame(umap)
umap$label <- label
colnames(umap)=c("UMAP1","UMAP2","label")
write.csv(umap,file="single_cell_umap.csv",row.names = F)

# umap <- read.table(file = "single_cell_own_output.txt",header=F)
# umap$label <- label$x
# colnames(umap)=c("UMAP1","UMAP2","label")
# write.csv(umap,file="single_cell_own_output.csv")
umap <- read.csv("single_cell_own_output.csv")
library(ggplot2)
g <- ggplot(umap,aes(x=UMAP1,y=UMAP2,color=label))
g <- g + geom_point()
g
