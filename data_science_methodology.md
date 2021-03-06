### Analytic Approach
- using Descriptive Analytics, study the past email history, and label the data
- using Diagnostic Analytics, it helps understand why these emails are marked as spam and moved to spam folder
- the two approaches above help understand the history data and label data properly
- I will classify the emails as spam and non-spam.
- the answer would be yes or no, we use the classification method to build the modelling
- if yes, the incoming email will be filtered out and move to spam, otherwise, it will be delivered to the inbox.

### Data Requirements
We will collect all emails from our email account, inlcuding indox, trash folder, spam folder, also sent folder in case the original email already removed from inbox
starting from these raw data for data requirement stage.

### Data Collection
The emails from our own accont is not enough, we also can download some sample emails from another email system. 
for example:
- from our other personal email accounts
- ask from friends etc.
we can also subscribe to some email notifications, promotions.
it's ok the email data format are different, and using different structures.

### Data Understanding and Preparation
- We understand the emails that always has following:
  - Misspelling in title
  - Emoticon in the title
  - Percentage number in the tile
  - Attachment with executable script hidden inside
  - Some hyperlink in the email body
  - The random generated sender email address
  etc, all these emails could be a sample of spam
  - when parsing the email body, we can have all these highlighted items as labelled variables, e.g. a column in final data frame, we can have 1 present yes, 0 present no in the table.
  
- Using plot to visualize the data, and help to understand the relationships between target vaiable and independent vaiables
  - find the meaningful columns that can determine the spam
  - get the maximum, minimum, mean values
  
- data washing and clean
  - delete some unuseful data
  - delete some unuseful columns
  - convert some text to numbers
  - remove the duplicated entries
  - finally create a flat table with all data merged in
  - normalize the data into a data frame

  
### Modeling and Evaluation
- we can use decision tree first, if the accurency doesn't reach the expectation, we can try another one.
- compare and evaluate the result to find the best method
  - split the data into 2 sets: training set and test set
  - compare the result and pick the best one

all these stages are iterative, we may need to go back to previous stage and discuss with client to further clear and clarify the problems.
