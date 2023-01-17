library(corrplot)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

### Figure 1
# Load the data for the correlation matrix
df <- read.csv('results/corr_mat_data_1.csv', sep = ',', check.names = FALSE)

# Compute the correlation matrix and p-values
M <- cor(df, use = 'pairwise.complete.obs')
testRes <- cor.mtest(df, conf.level = 0.95)

col_mat <- c(rep('red', 4), rep('blue', 4))

adjusted_pmat <- p.adjust(testRes$p, method='BH')
adjusted_pmat <- matrix(adjusted_pmat, 34, 34)
colnames(adjusted_pmat) <- colnames(testRes$p)
row.names(adjusted_pmat) <- row.names(testRes$p)

svg('figures/corr_mat_1.svg', width = 11, height = 11)
corrplot(M, p.mat = adjusted_pmat, method = 'pie', type = 'upper', diag = TRUE, 
         sig.level = 0.05, insig = 'blank', tl.cex=0.75, tl.col = c(rep('red', 5), rep('orange', 5), rep('blue', 4), 
                                                                    rep('blue', 4), rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('cadetblue4', 4), rep('cadetblue4', 4)), 
         cl.cex = 1.2)
segments(34.5, 29.5, -5, 29.5, col = 'black', lwd = 2)
segments(34.5, 24.5, -2, 24.5, col = 'black', lwd = 2)
segments(34.5, 20.5, 1, 20.5, col = 'black', lwd = 2)
segments(34.5, 16.5, 5, 16.5, col = 'black', lwd = 2)
segments(34.5, 12.5, 9, 12.5, col = 'black', lwd = 2)
segments(34.5, 8.5, 13, 8.5, col = 'black', lwd = 2)
segments(34.5, 4.5, 17, 4.5, col = 'black', lwd = 2)

segments(5.5, 56.5, 5.5, 29.5, col = 'black', lwd = 2)
segments(10.5, 56.5, 10.5, 24.5, col = 'black', lwd = 2)
segments(14.5, 56.5, 14.5, 20.5, col = 'black', lwd = 2)
segments(18.5, 56.5, 18.5, 16.5, col = 'black', lwd = 2)
segments(22.5, 56.5, 22.5, 12.5, col = 'black', lwd = 2)
segments(26.5, 56.5, 26.5, 8.5, col = 'black', lwd = 2)
segments(30.5, 56.5, 30.5, 4.5, col = 'black', lwd = 2)

text(-1.8, 31.5, expression(bold('Ab t'[1])), cex=1.0, srt = 90, font = 2)
text(0, 26.5, expression(bold('Ab t'[2])), cex=1.0, srt = 90, font = 2)
text(1.5, 22.5, expression(bold('CD3 t'[1])), cex=1.0, srt = 90, font = 2)
text(5.5, 18.5, expression(bold('CD4 t'[1])), cex=1.0, srt = 90, font = 2)
text(9.5, 14.5, expression(bold('CD8 t'[1])), cex=1.0, srt = 90, font = 2)
text(13.5, 10.5, expression(bold('CD3 t'[2])), cex=1.0, srt = 90, font = 2)
text(17.5, 6.5, expression(bold('CD4 t'[2])), cex=1.0, srt = 90, font = 2)
text(21.5, 2.75, expression(bold('CD8 t'[2])), cex=1.0, srt = 90, font = 2)

legend(1, 11.5, legend = c(expression('Serology at t'[1]), expression('Serology at t'[2]), expression('T-cell assays at t'[1]), 
                           expression('T-cell assays at t'[2])), 
       fill = c('red', 'orange', 'blue', 'cadetblue4'), cex=1.2)
dev.off()


### Figure 2
df <- read.csv('results/corr_mat_data_2.csv', sep = ',', check.names = FALSE)

# Compute the correlation matrix and p-values
M <- cor(df, use = 'pairwise.complete.obs')
testRes <- cor.mtest(df, conf.level = 0.95)

col_mat <- c(rep('red', 4), rep('blue', 4))

adjusted_pmat <- p.adjust(testRes$p, method='BH')
adjusted_pmat <- matrix(adjusted_pmat, 50, 50)
colnames(adjusted_pmat) <- colnames(testRes$p)
row.names(adjusted_pmat) <- row.names(testRes$p)

svg('figures/corr_mat_2.svg', width = 11, height = 11)
corrplot(M, p.mat = adjusted_pmat, method = 'pie', type = 'upper', diag = TRUE, 
         sig.level = 0.05, insig = 'blank', tl.cex=0.75, tl.col = c(rep('red', 5), rep('orange', 5), 
                                                                    rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('blue', 4), rep('cadetblue4', 4), 
                                                                    rep('blue', 4), rep('cadetblue4', 4),
                                                                    rep('blue', 4), rep('cadetblue4', 4),
                                                                    rep('blue', 4), rep('cadetblue4', 4)), cl.cex=1.2)
segments(50.5, 40.5, -8, 40.5, col = 'black', lwd = 2)
segments(50.5, 32.5, 2, 32.5, col = 'black', lwd = 2)
segments(50.5, 24.5, 10, 24.5, col = 'black', lwd = 2)
segments(50.5, 16.5, 17, 16.5, col = 'black', lwd = 2)
segments(50.5, 8.5, 26, 8.5, col = 'black', lwd = 2)

segments(10.5, 66.5, 10.5, 40.5, col = 'black', lwd = 2)
segments(18.5, 66.5, 18.5, 32.5, col = 'black', lwd = 2)
segments(26.5, 66.5, 26.5, 24.5, col = 'black', lwd = 2)
segments(34.5, 66.5, 34.5, 16.5, col = 'black', lwd = 2)
segments(42.5, 66.5, 42.5, 8.5, col = 'black', lwd = 2)

text(-3, 44.5, 'Ab', cex=1.0, srt = 90, font = 2)
text(-1, 36.5, 'CoV-Mix', cex=1.0, srt = 90, font = 2)
text(9, 28.5, 'N', cex=1.0, srt = 90, font = 2)
text(16, 20.5, 'S1', cex=1.0, srt = 90, font = 2)
text(24, 12.5, 'S', cex=1.0, srt = 90, font = 2)
text(32, 4.5, 'M', cex=1.0, srt = 90, font = 2)
dev.off()