# Clear the R-environment                                                               
rm(list = ls())
graphics.off()

# Set the working directory                                                              
setwd("K:/Data Science/Data Analysis/spam_ham_text_analysis/")

# Load the libraries                                                                    
library(ggplot2)
library(quanteda)
library(caret)
library(e1071)
library(irlba)
library(randomForest)

# Load the data                                                                             
spam_ham = read.csv('spam.csv', stringsAsFactors = FALSE)
spam_ham = spam_ham[,1:2]
names(spam_ham) = c('type', 'sms')

# About data                                                                                        
typeof(spam_ham)
str(spam_ham)
View(spam_ham)

# Check data to see if there are missing value                                                      
length(which(!complete.cases(spam_ham)))

# Fix the type of variables                                                                         
typeof(spam_ham$type)
typeof(spam_ham$sms)
spam_ham$type = as.factor(spam_ham$type)

# Explore the data_________(distribution of lebels)                                                 
table(spam_ham$type)
prop.table(table(spam_ham$type))

# Distribution of text length                                                                       
spam_ham$sms_length = nchar(spam_ham$sms)
summary(spam_ham$sms_length)

# Visualization of distributions                                                                    
p1 = ggplot(spam_ham, aes(x = type, y = sms_length, color = type)) + 
  geom_boxplot() +
  labs(title = "Boxplot of Type of SMS & SMS Length",
       x = "Type of SMS", y = "SMS Length") +
  theme_bw()
p1

p2 = ggplot(spam_ham, aes(x = sms_length, fill = type)) +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of SMS Lengths with Type of SMS",
       x = "SMS Lengths", y = "SMS Count") +
  theme_bw()
p2

# Data Partition                                                
set.seed(32984)
indexes = createDataPartition(spam_ham$type, times = 1, p = 0.7, list = FALSE)

train = spam_ham[indexes,]
test = spam_ham[-indexes,]

# Check the proportion                                          
prop.table(table(train$type))
prop.table(table(test$type))

help(package = 'quanteda')

# Tokonization                                                  
train_tokens = tokens(train$sms, what = 'word',
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, split_hyphens = TRUE)
train_tokens[[5]]

## Lower case of tokens                                         
train_tokens = tokens_tolower(train_tokens)

train_tokens[[5]]

## Remove english stopwords                                     
quanteda::stopwords()
train_tokens = tokens_select(train_tokens, stopwords(),
                             selection = 'remove')
train_tokens[[5]]

# Stemming (Collapse down the simmilar word)                    
train_tokens = tokens_wordstem(train_tokens, language = 'english')
?quanteda::tokens_wordstem

train_tokens[[5]]

train_token_df = cbind(type = train$type, as.data.frame(train_tokens_dfm))
names(train_token_df) = make.names(names(train_token_df))
set.seed(48753)
cv_folds = createMultiFolds(train$type, k = 10, times = 3)
cv_control = trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv_folds)

library(doSNOW)
start_time = Sys.time()
cl <- makeCluster()







# Create DFM or BOW(Bag of Word) Model                          
train_tokens_dfm = dfm(train_tokens)
train_tokens_matrix = as.matrix(train_tokens_dfm)
dim(train_tokens_matrix)
View(train_tokens_matrix[1:100,100:200])

# Investigate the effects of stemming                           
colnames(train_tokens_matrix)[1:50]

# TF (Term Frequency)
tf = function(row){
  row/sum(row)
}

idf = function(col){
  corpus_size = length(col)
  doc_count = length(which(col > 0))
  
  log10(corpus_size/doc_count)
}

tf_idf = function(tf, idf){
  tf * idf
}

train_tokens_tf = apply(train_tokens_matrix, 1, tf)
train_tokens_idf = apply(train_tokens_matrix, 2, idf)
train_tokens_tf_idf = apply(train_tokens_tf, 2, tf_idf, idf = train_tokens_idf)
View(train_tokens_tf_idf)

# Transpose the matrix                                                  
train_tokens_tf_idf = t(train_tokens_tf_idf)

# Check for incomplete cases                                            
incomplete_cases = which(!complete.cases(train_tokens_tf_idf))
train$sms[incomplete_cases]

# Fix incomplete cases                                                  
train_tokens_tf_idf[incomplete_cases,] = rep(0.0, ncol(train_tokens_tf_idf))
sum(which(!complete.cases(train_tokens_tf_idf)))

# Clean data frame                                                      
train_tokens_tf_idf_df = cbind(type = train$type, data.frame(train_tokens_tf_idf))
names(train_tokens_tf_idf_df) = make.names(names(train_tokens_tf_idf))
View(train_tokens_tf_idf_df)


# Ngram