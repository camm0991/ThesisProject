# Notes about datasets

- Two datasets where made for testing and training, each of them is composed by five sampling experiments where 1000 records per class were taken with a duration of 7.5 seconds. The datasets where sorted by class but the order of the records were maintained. The classes correspond to the following intentions:
-- 0 -> Fixed position
-- 1 -> Go forward
-- 2 -> Go backwards
-- 3 -> Move to the left
-- 4 -> Move to the right
- The filtered version of the dataset is also included, a fourth class low pass filter was applied to both datasets in a range between 30Hz and 128Hz.
- Plotting of the filtered (F) and non-filtered (NF) signals of the training dataset are included.