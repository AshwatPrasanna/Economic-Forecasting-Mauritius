How does this model work?
First, the existing data has been grouped into several sheets. Each sheet is forecasted using TCN (which is suitable for periodic/repeating data with changing trends) and VAR on the residuals
- Each of these sheets has been scaled using a StandardScaler, which is defined based only on the static, unchanging data (and will not change with further testing or testing variations)
- All of the columns' scores are combined using an ANN, year after year after year