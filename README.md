# Machine Learning-Based Classification Of Countries Based On Food Supply Quantities In The Caucasus And Surrounding Regions

Please cite this study if you use it in your research as follows:

Duman, Hakan. 2024. “Machine Learning-Based Classification Of Countries
Based On Food Supply Quantities In The Caucasus And Surrounding
Regions.” In Kafkasya Araştırmaları - II, edited by Oğuz Şimşek, Çağrı
Akgün, and Çetin İzgi, 189–208. Ankara: Sonçağ Akademi.

[Full text of the study](https://www.researchgate.net/publication/387441081_MACHINE_LEARNING-BASED_CLASSIFICATION_OF0ACOUNTRIES_BASED_ON_FOOD_SUPPLY_QUANTITIES_IN0ATHE_CAUCASUS_AND_SURROUNDING_REGIONS)

## Abstract

Food security is vital for human survival, ensuring consistent access to
sufficient, safe, and nutritious food for an active, healthy life. This
study addresses food security challenges in the Caucasus region by
analyzing FAOSTAT food balance data (2010–2022) on daily per capita food
supply (kcal/capita/day) for plant- and animal-based products. Using R
and Python, statistical analyses (Welch ANOVA and Games-Howell tests)
and machine learning models (Logistic Regression, Random Forest,
Decision Tree, and Multi-Layer Perceptron) were applied, with 256
hyperparameter combinations and a CRS-DEA model assessing algorithm
efficiency. Results revealed significant variations in food consumption
patterns, with Kazakhstan excelling in animal product intake and Türkiye
leading in vegetal products, while Ukraine’s food supply declined due to
conflict. Decision Tree (DT) emerged as the most suitable machine
learning model, balancing high performance, minimal computational time,
and interpretability. These findings contribute valuable insights into
food security and machine learning model efficiency, providing a
foundation for future research and practical applications.

Keywords:\*\* Food Security, Machine Learning Models, Consumption
Patterns, Caucasus Region, Decision Tree Analysis

## R Packages

R programming language (version 4.2.2) ([Anonymous, 2022](#ref-r_2022))
and several key packages were used in this study. The tidyverse package
(version 2.0.0) supported data cleaning and visualization ([Wickham et
al., 2019a](#ref-wickham_welcome_2019)), while the rsample package
(version 1.2.0) facilitated resampling ([Frick et al.,
2023](#ref-frick_rsample_2023)). The reticulate package (version 1.28)
enabled Python integration ([Ushey, Allaire, & Tang,
2023](#ref-ushey_reticulate_2023)), and the Benchmarking package
(version 0.32) was used for DEA methods ([Bogetoft & Otto,
2024](#ref-benchmarks_dea_sfa_2024)). Additionally, we used the rstatix
package (version 0.7.2) ([Kassambara, 2023](#ref-rstatix_2023)) for
statistical tests.

Although R is widely recognized for its statistical capabilities, its
computational speed can be limiting. To address this, we employed Python
(version 3.13.0) and the scikit-learn library (version 1.5.2)
([Pedregosa et al., 2011](#ref-scikit-learn)) to train models more
efficiently. The source code for the study is available in the
associated GitHub repository[1].

## Acknowledgements

This analysis adapted and modified code from various sources, such as
books, package manuals, vignettes, and GitHub repositories. The sources
are cited as follows:

-   Data exploring, preparing, manipulation, cleaning, and
    visualization: Wickham et al. ([2019b](#ref-tidyverse-2019)), Wang,
    Cook, & Hyndman ([2020](#ref-tsibble-2020)), Wang & contibutors
    ([2024](#ref-tsibble-2024-github)), Wickham & contibutors
    ([2024](#ref-ggplot2-2024-github)), Wickham
    ([2016](#ref-ggplot2-2016)), Wickham, Hester, & Bryan
    ([2024](#ref-readr-2024-github)), Wickham, Vaughan, & Girlich
    ([2024](#ref-tidyr-2024-github)), Slowikowski
    ([2023](#ref-slowikowski_2023)), Xie ([2023](#ref-xie_knitr_2023)),
    Xie ([2015](#ref-xie_knitr_2015)), Xie
    ([2014](#ref-xie_knitr_2014)), Cui
    ([2020](#ref-cui_dataexplorer_2020)), Wilke
    ([2020](#ref-wilke_cowplot_2020)), Grolemund & Wickham
    ([2011](#ref-grolemund_hadley_2011))  
-   Map Visualization: Massicotte & South
    ([2023](#ref-rnaturalearth-2023)), South
    ([2017](#ref-rnaturalearthdata-2017)), Pebesma & Bivand
    ([2005](#ref-rnews-2005)), Bivand, Pebesma, & Gomez-Rubio
    ([2013](#ref-asdar-2013)), Pebesma & contibutors
    ([2024](#ref-sf-2024-github))
-   Training Models: Frick et al. ([2023](#ref-frick_rsample_2023)),
    Ushey et al. ([2023](#ref-ushey_reticulate_2023)),
-   Statistical Tests: Kassambara ([2023](#ref-rstatix_2023)), Fox &
    Weisberg ([2019](#ref-fox_r_2019)), Bogetoft & Otto
    ([2024](#ref-benchmarks_dea_sfa_2024))

## Code References

Anonymous. (2022). *R: A language and environment for statistical
computing*. Vienna, Austria: R Foundation for Statistical Computing.
Retrieved from <https://www.R-project.org/>

Bivand, R. S., Pebesma, E. J., & Gomez-Rubio, V. (2013). *<span
class="nocase">Applied spatial data analysis with R, Second
edition</span>*. Springer, NY. Retrieved from <https://asdar-book.org/>

Bogetoft, P., & Otto, L. (2024). *Benchmarking with DEA and SFA*.

Cui, B. (2020). *DataExplorer: Automate data exploration and treatment*.
Retrieved from <https://CRAN.R-project.org/package=DataExplorer>

Fox, J., & Weisberg, S. (2019). *<span class="nocase">An R Companion to
Applied Regression</span>*. London: SAGE.

Frick, H., Chow, F., Kuhn, M., Mahoney, M., Silge, J., & Wickham, H.
(2023). *Rsample: General resampling infrastructure*. Retrieved from
<https://CRAN.R-project.org/package=rsample>

Grolemund, G., & Wickham, H. (2011). Dates and times made easy with
<span class="nocase">lubridate</span>. *Journal of Statistical
Software*, *40*(3), 1–25. Retrieved from
<https://www.jstatsoft.org/v40/i03/>

Kassambara, A. (2023). *Rstatix: Pipe-friendly framework for basic
statistical tests*. Retrieved from
<https://CRAN.R-project.org/package=rstatix>

Massicotte, P., & South, A. (2023). *<span class="nocase">rnaturalearth:
World Map Data from Natural Earth</span>*. Retrieved from
<https://CRAN.R-project.org/package=rnaturalearth>

Pebesma, E. J., & Bivand, R. (2005). <span class="nocase">Classes and
methods for spatial data in R</span>. *R News*, *5*(2), 9–13. Retrieved
from <https://CRAN.R-project.org/doc/Rnews/>

Pebesma, E. J., & contibutors. (2024). <span class="nocase">Simple
features for R</span>. Retrieved June 2, 2025, from
<https://r-spatial.github.io/sf/>

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
Grisel, O., … Duchesnay, E. (2011). Scikit-learn: Machine learning in
Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Slowikowski, K. (2023). *Ggrepel: Automatically position non-overlapping
text labels with ’ggplot2’*. Retrieved from
<https://CRAN.R-project.org/package=ggrepel>

South, A. (2017). *<span class="nocase">rnaturalearthdata: World Vector
Map Data from Natural Earth Used in ’rnaturalearth’</span>*. Retrieved
from <https://CRAN.R-project.org/package=rnaturalearthdata>

Ushey, K., Allaire, J., & Tang, Y. (2023). *Reticulate: Interface to
’python’*. Retrieved from
<https://CRAN.R-project.org/package=reticulate>

Wang, E., & contibutors. (2024). <span
class="nocase">tidyverts/tsibble</span>. Retrieved June 2, 2025, from
<https://github.com/tidyverts/tsibble>

Wang, E., Cook, D., & Hyndman, R. J. (2020). <span class="nocase">A new
tidy data structure to support exploration and modeling of temporal
data</span>. *Journal of Computational and Graphical Statistics*,
*29*(3), 466–478. <https://doi.org/10.1080/10618600.2019.1695624>

Wickham, H. (2016). *<span class="nocase">ggplot2: Elegant Graphics for
Data Analysis</span>*. Springer-Verlag New York. Retrieved from
<https://ggplot2.tidyverse.org>

Wickham, H., Averick, M., Bryan, J., Chang, W., McGowan, L. D.,
François, R., … Yutani, H. (2019a). <span class="nocase">Welcome to the
tidyverse</span>. *Journal of Open Source Software*, *4*(43), 1686.
<https://doi.org/10.21105/joss.01686>

Wickham, H., Averick, M., Bryan, J., Chang, W., McGowan, L. D.,
François, R., … Yutani, H. (2019b). <span class="nocase">Welcome to the
<span class="nocase">tidyverse</span></span>. *Journal of Open Source
Software*, *4*(43), 1686. <https://doi.org/10.21105/joss.01686>

Wickham, H., & contibutors. (2024). <span
class="nocase">tidyverse/ggplot2</span>. Retrieved June 2, 2025, from
<https://github.com/tidyverse/ggplot2>

Wickham, H., Hester, J., & Bryan, J. (2024). *<span
class="nocase">readr: Read Rectangular Text Data</span>*. Retrieved from
<https://readr.tidyverse.org>

Wickham, H., Vaughan, D., & Girlich, M. (2024). *<span
class="nocase">tidyr: Tidy Messy Data</span>*. Retrieved from
<https://tidyr.tidyverse.org>

Wilke, C. O. (2020). *Cowplot: Streamlined plot theme and plot
annotations for ’ggplot2’*. Retrieved from
<https://CRAN.R-project.org/package=cowplot>

Xie, Y. (2014). Knitr: A comprehensive tool for reproducible research in
R. In V. Stodden, F. Leisch, & R. D. Peng (Eds.), *Implementing
reproducible computational research*. Chapman; Hall/CRC.

Xie, Y. (2015). *Dynamic documents with R and knitr* (2nd ed.). Boca
Raton, Florida: Chapman; Hall/CRC. Retrieved from
<https://yihui.org/knitr/>

Xie, Y. (2023). *Knitr: A general-purpose package for dynamic report
generation in r*. Retrieved from <https://yihui.org/knitr/>

[1] <https://github.com/hakan-duman-acad/chapter-foodbalance-ml-classification>
