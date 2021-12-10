library(corrplot)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

### Figure 1
# Load the data for the correlation matrix
df <- read.csv('./corr_mat_data_1.csv', sep = ',', check.names = FALSE)

# Compute the correlation matrix and p-values
M <- cor(df, use = 'complete.obs')
testRes <- cor.mtest(df, conf.level = 0.95)

col_mat <- c(rep('red', 4), rep('blue', 4))

adjusted_pmat <- p.adjust(testRes$p, method='BH')
adjusted_pmat <- matrix(adjusted_pmat, 32, 32)
colnames(adjusted_pmat) <- colnames(testRes$p)
row.names(adjusted_pmat) <- row.names(testRes$p)

svg('figures/corr_mat_1.svg', width = 11, height = 11)
corrplot(M, p.mat = adjusted_pmat, method = 'pie', type = 'upper', diag = TRUE, 
         sig.level = 0.05, insig = 'blank', tl.cex=0.75, tl.col = c(rep('red', 4), rep('orange', 4), rep('blue', 4), 
                                                                    rep('blue', 4), rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('cadetblue4', 4), rep('cadetblue4', 4)), 
         cl.cex = 1.2)
segments(32.5, 28.5, -5, 28.5, col = 'black', lwd = 2)
segments(32.5, 24.5, -2, 24.5, col = 'black', lwd = 2)
segments(32.5, 20.5, 1, 20.5, col = 'black', lwd = 2)
segments(32.5, 16.5, 5, 16.5, col = 'black', lwd = 2)
segments(32.5, 12.5, 9, 12.5, col = 'black', lwd = 2)
segments(32.5, 8.5, 13, 8.5, col = 'black', lwd = 2)
segments(32.5, 4.5, 17, 4.5, col = 'black', lwd = 2)

segments(4.5, 56.5, 4.5, 28.5, col = 'black', lwd = 2)
segments(8.5, 56.5, 8.5, 24.5, col = 'black', lwd = 2)
segments(12.5, 56.5, 12.5, 20.5, col = 'black', lwd = 2)
segments(16.5, 56.5, 16.5, 16.5, col = 'black', lwd = 2)
segments(20.5, 56.5, 20.5, 12.5, col = 'black', lwd = 2)
segments(24.5, 56.5, 24.5, 8.5, col = 'black', lwd = 2)
segments(28.5, 56.5, 28.5, 4.5, col = 'black', lwd = 2)

text(-1.8, 30.5, 'IgG t1', cex=1.0, srt = 90, font = 2)
text(0, 26.5, 'IgG t2', cex=1.0, srt = 90, font = 2)
text(1.5, 22.5, 'CD3 t1', cex=1.0, srt = 90, font = 2)
text(5.5, 18.5, 'CD4 t1', cex=1.0, srt = 90, font = 2)
text(9.5, 14.5, 'CD8 t1', cex=1.0, srt = 90, font = 2)
text(13.5, 10.5, 'CD3 t2', cex=1.0, srt = 90, font = 2)
text(17.5, 6.5, 'CD4 t2', cex=1.0, srt = 90, font = 2)
text(21.5, 2.75, 'CD8 t2', cex=1.0, srt = 90, font = 2)

legend(1, 11.5, legend = c('Serology at t1', 'Serology at t2', 'T-cell assays at t1', 'T-cell assays at t2'), 
       fill = c('red', 'orange', 'blue', 'cadetblue4'), cex=1.2)
dev.off()


### Figure 2
df <- read.csv('./corr_mat_data_2.csv', sep = ',', check.names = FALSE)

# Compute the correlation matrix and p-values
M <- cor(df, use = 'complete.obs')
testRes <- cor.mtest(df, conf.level = 0.95)

col_mat <- c(rep('red', 4), rep('blue', 4))

adjusted_pmat <- p.adjust(testRes$p, method='BH')
adjusted_pmat <- matrix(adjusted_pmat, 48, 48)
colnames(adjusted_pmat) <- colnames(testRes$p)
row.names(adjusted_pmat) <- row.names(testRes$p)

svg('figures/corr_mat_2.svg', width = 11, height = 11)
corrplot(M, p.mat = adjusted_pmat, method = 'pie', type = 'upper', diag = TRUE, 
         sig.level = 0.05, insig = 'blank', tl.cex=0.75, tl.col = c(rep('red', 4), rep('orange', 4), 
                                                                    rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('blue', 4), rep('cadetblue4', 4),
                                                                    rep('blue', 4), rep('cadetblue4', 4),
                                                                    rep('blue', 4), rep('cadetblue4', 4)), cl.cex=1.2)
segments(48.5, 40.5, -8, 40.5, col = 'black', lwd = 2)
segments(48.5, 32.5, 2, 32.5, col = 'black', lwd = 2)
segments(48.5, 24.5, 10, 24.5, col = 'black', lwd = 2)
segments(48.5, 16.5, 17, 16.5, col = 'black', lwd = 2)
segments(48.5, 8.5, 26, 8.5, col = 'black', lwd = 2)

segments(8.5, 66.5, 8.5, 40.5, col = 'black', lwd = 2)
segments(16.5, 66.5, 16.5, 32.5, col = 'black', lwd = 2)
segments(24.5, 66.5, 24.5, 24.5, col = 'black', lwd = 2)
segments(32.5, 66.5, 32.5, 16.5, col = 'black', lwd = 2)
segments(40.5, 66.5, 40.5, 8.5, col = 'black', lwd = 2)

text(-3, 44.5, 'IgG', cex=1.0, srt = 90, font = 2)
text(-1, 36.5, 'Cov-mix', cex=1.0, srt = 90, font = 2)
text(9, 28.5, 'N', cex=1.0, srt = 90, font = 2)
text(16, 20.5, 'S1', cex=1.0, srt = 90, font = 2)
text(24, 12.5, 'S', cex=1.0, srt = 90, font = 2)
text(32, 4.5, 'M', cex=1.0, srt = 90, font = 2)
dev.off()