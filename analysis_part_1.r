list.of.packages <- c("tidyverse", "rsample","reticulate","lubridate","ggrepel",
                      "rstatix","agricolae", "car","knitr","DataExplorer",
                     "rnaturalearth","rnaturalearthdata","sf","cowplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

for (i in list.of.packages) {
    library(i, ,character.only = TRUE)
}

data <- read_csv("./data/data.csv")
data2 <- data |> select(Year,Area, Item, Value) |>
        filter(Area %in% c("Türkiye","Azerbaijan","Armenia","Russian Federation",
                           "Georgia","Iran (Islamic Republic of)","Ukraine","Kazakhstan")) |>
    mutate(Area = if_else(Area == "Iran (Islamic Republic of)","Iran",Area),
          Area = if_else(Area == "Russian Federation","Russia",Area)) |>
      rename("Country" = Area) 
dat <- data2 |> 
    pivot_wider(names_from = c("Item"), values_from = "Value") |>
    select(-`Grand Total`) |> 
    ungroup() |>
    mutate(Year = ymd(paste0(Year,"-01-01")))
dat |> sample_n(5)
dat |> write_csv("./data/data_cleaned.csv")

data2 <- dat |>
  pivot_longer(cols = 3:4, names_to = "Item", values_to = "Value")
data3 <- dat |> rename(VP=3,AP=4) 
data3 |> sample_n(5)

leveneTest(VP~Country,data3)
leveneTest(AP~Country,data3)

one_way_2 <- function (y="AP",x="Country",data = data3) {
data <- data |> rename(y = {{y}}, x = {{x}})
model <- oneway.test(y~x, data = data, var.equal = FALSE)
print(model)
mns <- data |>
    group_by(x) |>
    summarise(mean = mean(y), sd = sd(y)) |>
    arrange(desc(mean))
lst <- games_howell_test(data, formula=y~x, 
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

one_way_2(y="VP")

one_way_2("AP")

dat |> pivot_longer(cols = 3:4,names_to = "Variable",values_to = "Values") |>
    ggplot(aes(x=Country, y= Values)) +
    geom_boxplot(fill = "lightblue", color = "black", outlier.shape = 16)+
    facet_wrap(~Variable, ncol=2,  scales="free_y")+
    cowplot::theme_cowplot()+ 
    theme(axis.text.x = element_text(angle = 90, vjust = -0.0, 
                                     hjust=-.0))+
    ylab("Food Supply (kcal/capita/day)") + xlab("")+
  theme(legend.position = "none")
dat |> pivot_longer(cols = 3:4,names_to = "Variable",values_to = "Values") |>
    ggplot(aes(x=Year, y= Values)) +
    geom_line(aes(color=Country))+
    facet_wrap(~Variable, ncol=2,  scales="free_y")+
    cowplot::theme_cowplot()+ 
    theme(axis.text.x = element_text(angle = 90, vjust = -0.0, 
                                     hjust=-.0))+
    ylab("Food Supply (kcal/capita/day)") + xlab("")+
  theme(legend.position = "bottom")

dat <- dat |> rename(VP=3,AP=4) |> select(-Year)
introduce(dat)

asia <- ne_countries(country = c("turkey","russia","georgia",
                                 "armenia","azerbaijan",
                                "kazakhstan","ukraine","iran"), 
                          scale = "large",
                          returnclass = "sf") |>
    mutate(admin = if_else(admin == "Turkey", "Türkiye",admin))
asiamap <- ggplot(asia) +
  geom_sf(aes(fill=admin)) +
  theme_minimal() +
  coord_sf(crs = "+proj=laea +lat_0=0 +lon_0=0")
asiamap +
    theme(legend.position = "right")+
    labs(fill="Countries")

asiamap <- asia |> left_join(dat |> group_by(Country) |>
    summarise(value =mean(AP)), by = c("admin" = "Country")) |>
ggplot() +
  geom_sf(aes(fill = value)) +
  theme_minimal() +
  coord_sf(crs = "+proj=laea +lat_0=0 +lon_0=0")
asiamap +
    geom_sf_text(aes(label = admin), size = 3,nudge_x = 0.1)+
    labs(fill = "AP") + xlab("")+ylab("")


asiamap <- asia |> left_join(dat |> group_by(Country) |>
    summarise(value =mean(VP)), by = c("admin" = "Country")) |>
ggplot() +
  geom_sf(aes(fill = value)) +
  theme_minimal() +
  coord_sf(crs = "+proj=laea +lat_0=0 +lon_0=0")
asiamap +
    geom_sf_text(aes(label = admin), size = 3,nudge_x = 0.1,color="grey")+
    labs(fill = "VP") + xlab("")+ylab("")

env_name <- "r-reticulate"
use_condaenv(env_name, required = TRUE)
required_packages <- c("memory_profiler==0.61.0", 
                       "numpy==1.24.3", 
                       "numpy-base==1.24.3", 
                       "numpydoc==1.5.0", 
                       "pandas==2.2.3", 
                       "scikit-learn==1.5.2")

# Python ortamında yüklü paketleri kontrol etme
installed_packages <- py_list_packages()
installed_packages <- installed_packages$requirement |> str_split("=",simplify = TRUE) 
installed_packages <- installed_packages |> apply(1, function(x) paste0(x, collapse= "=="))
# Yüklenmemiş paketleri belirleme
missing_packages <- required_packages[!(required_packages %in% installed_packages)]

# Eksik paketleri yükleme
if (length(missing_packages) > 0) {
  py_install(missing_packages, envname = env_name)
  cat("Missing packages installed:\n", paste(missing_packages, collapse = "\n"))
} else {
  cat("All packages already installed.\n")
}
source_python("model.py")

models <- c("LR","DT","RF","MLP")
seeds <- round(seq(100,1000,length.out = 25),0)

mdls2 <- NULL
sink("parameters.txt")
for (i in seeds) {
    set.seed(i)
    splits <- initial_split(dat, strata = Country, prop = .80)
    tr_data <- training(splits)
    ts_data  <- testing(splits) 
    
    X_train <- tr_data |> 
                select(-Country) 
    y_train <- tr_data |>  
                select(Country) |> 
                unlist() |> as.vector()
    X_test <- ts_data |> 
                select(-Country)
    y_test <- ts_data |>
        select(Country) |>
        unlist() |>
        as.vector()

    for (j in models){
    mdl <- Model(X = X_train, y = y_train, model = j, 
                 cv =3, seed = i)
    mdl$train()
    vals <- mdl$get_all_info(X_test,y_test)
    tmp <- tibble( seed = i, model_name = j,
                    model =  list(mdl),
                   time = vals[1], 
                   train_score= vals[3], 
                      test_score= vals[4])
    mdls2 <- rbind(mdls2,tmp)    
     cat("Seed :", i, "Model :",j,"Parameters :" , 
         unlist(mdl$get_best_params()[-1]),"\n")
    }}
sink()

mdls2 |> write_rds("./data/cau_results.Rds",compress = "bz")
