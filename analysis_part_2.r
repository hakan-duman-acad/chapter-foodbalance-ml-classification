list.of.packages <- c("tidyverse", "reticulate","lubridate","ggrepel","Benchmarking",
                      "rstatix","agricolae", "car","knitr","cowplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

for (i in list.of.packages) {
    library(i, ,character.only = TRUE)
}
one_way_2 <- function (y="AP",x="Country",data) {
data <- data |> rename(y = {{y}}, x = {{x}})
data <- data |> mutate(x = factor(x))
print(leveneTest(y ~ x , data = data))

#     print(data)
model <- oneway.test(y~x, data = data, var.equal = FALSE)
# summary(model)
print(model)
mns <- data |>
    group_by(x) |>
    summarise(mean = mean(y), sd = sd(y)) |>
    arrange(desc(mean))
lst <- rstatix::games_howell_test(data, formula=y~x, 
                         conf.level = 0.95, detailed = FALSE) 
a <- lst |>  mutate(gr1 = factor (group1, levels = mns$x),
              gr2 = factor (group2, levels = mns$x)) |>
filter(p.adj.signif=="ns") |> select(gr1,gr2) 
a1 <- a
colnames(a1) <- c("gr2","gr1")
tbl <- rbind(a,a1) |> table()

mns <- mns |> mutate(l = "")
tbl
lidx <- 1
excl <- NULL
for (i in 1:nrow(tbl)) {
    nl <- letters[lidx]
    c1 <- rownames(tbl)[i]
    if (i %in% excl) next
    if (sum(tbl[i,])==0) {
            mns$l[i] = paste0(mns$l[i],nl)
            lidx <- lidx +1
            excl <- c(excl,i)
        } else {
          idx<- which(tbl[i,]==1)
        mns$l[i] = paste0(mns$l[i],nl)
        mns$l[idx] = paste0(mns$l[idx],nl)
            excl <- c(excl,i, idx)
                lidx <- lidx +1
        }}
return(mns)
}
options(scipen = 99, digits = 2)

data <- read_rds(file = "./data/cau_results.Rds")
data |> arrange(desc(test_score)) 

one_way_2(y = test_score,x = model_name, data=data)|> 
    mutate(mean = paste0(round(mean,2),l,"+-",round(sd,2))) |>
    select(-sd, -l) |>
    kable(format = "markdown", booktabs = TRUE, 
          col.names = c("Model","Score"))

one_way_2(y = train_score,x = model_name, data=data)|> 
    mutate(mean = paste0(round(mean,2),l,"+-",round(sd,2))) |>
    select(-sd, -l) |>
    kable(format = "markdown", booktabs = TRUE, 
          col.names = c("Model","Score"))

one_way_2(y = time,x = model_name, data=data)|> 
    mutate(mean = paste0(round(mean,2),l,"+-",round(sd,2))) |>
    select(-sd, -l) |>
    kable(format = "markdown", booktabs = TRUE, 
          col.names = c("Model","Time"))

data2 <- data |>
    group_by(model_name) |>
    summarise(score=mean(test_score), time = mean(time)) 
data3 <- data2 |> ungroup()
inputs <- as.matrix(data3 |> select(time))
outputs <- as.matrix(data3 |> select(score))
nms <- paste0(data2$model_name)
result <- dea(inputs, outputs, RTS = "crs")  
print(data.frame(model = nms, Efficieny = result$eff) |> arrange(desc(Efficieny))) |> kable("markdown") # Etkinlik skorlarını görüntüleme
summary(result)
