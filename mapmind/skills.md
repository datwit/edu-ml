DataSci Skills
******************

Theoretical
==============

* Creation of visualizations to explain the results.
* Storytelling
* Formulation of questions and preparation/testing of hypotheses.
* Domain knowledge.
* BigData
* NLP, NN & DL

Maths, Probability, Stats
============================

* Have a good foundation in algebra, calculus, probability, and statistics 
(the maths that we swallow in the first 2 courses of any career in 
engineering).

* __Bayes’s theorem__. It’s a foundational pillar of probability theory, and
it comes up all the time in interviews. You should practice doing some
basic Bayes theorem whiteboarding problems, and read the first chapter
of this famous book to get a rock-solid understanding of the origin and
meaning of the rule (bonus: it’s actually a fun read!).

* __Basic probability__. You should be able to answer questions like these.

* __Model evaluation__. In classification problems, for example, most n00bs
default to using model accuracy as their metric, which is usually a
terrible choice. Get comfortable with sklearn's precision_score,
recall_score, f1_score , and roc_auc_score functions, and the theory
behind them. For regression tasks, understanding why you would use
mean_squared_error rather than mean_absolute_error (and vice-versa) is
also crucial. It’s really worth taking the time to check out all the
model evaluation metrics listed in sklearn's official documentation.

Practical ML
================

* Machine Learning (algorithms, modeling, evaluation, optimization, etc).

	- [Course of Jeremy Howard](https://twitter.com/jeremyphoward)

* __Data exploration__. You should have pandas functions like .corr(),
scatter_matrix() , .hist() and .bar() on the tip of your tongue. You
should always be looking for opportunities to visualize your data using
PCA or t-SNE, using sklearn's PCA and TSNE functions.

* __Feature selection__. 90% of the time, your dataset will have way more
features than you need (which leads to excessive training time, and a
heightened risk of overfitting). Get familiar with basic filter methods
(look up scikit-learn’s VarianceThreshold and SelectKBest functions),
and more sophisticated model-based feature selection methods (look up
SelectFromModel).

* __Hyperparameter search for model optimization__. You definitely should
know what GridSearchCV does and how it works. Likewise for
RandomSearchCV. To really stand out, try experimenting with skopt's
BayesSearchCV to learn how you can apply bayesian optimization to your
hyperparameter search.

* __Pipelines__. Use sklearn's pipeline library to wrap their preprocessing,
feature selection and modeling steps together. Discomfort with pipeline
is a huge tell that a data scientist needs to get more familiar with
their modeling toolkit.

* Knowledge of SQL to make queries about databases (with joinsand those things, not difficult).
* Obtaining data from different sources (API queries, web scrapping, …).


* Deep Learning, Reinforcement Learning, Natural Language Processing, Computer Vision, …

	- [DL Coursera](https://www.coursera.org/specializations/deep-learning)

Software Engineering
===========

* Python or R as a programming language, and their corresponding libraries for Data Science.
	- [Exercises](https://www.hackerrank.com/domains/tutorials/30-days-of-code)
* Industrialization (AWS/GCloud), Goku

* __Version control__. You should know how to use git , and interact with your
remote GitHub repos using the command line. If you don’t, I suggest
starting with this tutorial. 
* __Web development__. Some companies like their
data scientists to be comfortable accessing data that’s stored on their
web app, or via an API. Getting comfortable with the basics of web
development is important, and the best way to do that is to learn a bit
of Flask. 
* __Web scraping__. Sort of related to web development: sometimes,
you’ll need to automate data collection by scraping data from live
websites. Two great tools to consider for this are BeautifulSoup and
scrapy. 
* __Clean code__. Learn how to use docstrings. Don’t overuse inline
comments. Break your functions up into smaller functions. Way smaller.
There shouldn’t be functions in your code longer than 10 lines of code.
Give your functions good, descriptive names ( function_1 is not a good
name). Follow pythonic convention and name your variables with
underscores like_this and not LikeThis or likeThis . Don’t write python
modules ( .py files) with more than 400 lines of code. Each module
should have a clear purpose (e.g. data_processing.py, predict.py ).
Learn what an if name == '__main__': code block does and why it’s
important. Use list comprehension. Don’t over-use for loops. Add a
README file to your project.

business instinct
===================

An alarming number of people seem to think that getting hired is about
showing that you’re the most technically competent applicant to a role.
It’s not. In reality, companies want to hire people who can help them
make more money, faster.

In general that means moving beyond just technical ability, and
building a number of additional skills:

* Making something people want. When most people are in “data science
learning mode”, they follow a very predictable series of steps: import
data, explore data, clean data, visualize data, model data, evaluate
model. And that’s fine when you’re focused on learning a new library or
technique, but going on autopilot is a really bad habit in a business
environment, where everything you do costs the company time (money).
You’ll want to get good at thinking like a business, and making good
guesses as to how you can best leverage your time to make meaningful
contributions to your team and company. A great way to do this is to
decide on some questions that you want your data science projects to
answer before you begin them (so that you don’t get carried away with
irrelevant tasks that form part of the otherwise “standard” DS
workflow). Make these questions as practical as possible, and after
you’ve completed your project, reflect on how well you were able to
answer them.

* Asking the right questions. Companies want to hire people who are able
to keep the big picture in mind while they tune their models, and ask
themselves questions like, “am I building this because it’s going to be
legitimately helpful to my team and company, or because it’s a cool use
case for an algorithm I really like?” and “what key business metric am
I trying to optimize, and is there a better way to do that?”.

* Explaining your results. Management needs you to tell them what
products are selling well, or which users are leaving for a competitor
and why, but they have no idea (and don’t care about) what a
precision/recall curve is, or how hard it was for you to avoid
overfitting your model. For that reason, a key skill is the ability to
convey your results and their implications to nontechnical audiences.
Try building a project and explaining it to a friend who hasn’t taken
math since high school (hint: your explanation shouldn’t involve any
algorithm names, or refer to hyperparameter tuning. Simple words are
better words.).

Others
=========

* Agile

# TODO: Develop this field of jobs interview searching and ziping more
internet web articles.

# TODO: extract the skills from this URLS

* https://www.kdnuggets.com/2020/01/resources-become-data-engineer.html
* https://www.kdnuggets.com/2020/01/top-5-data-science-skills-2020.html
* wwww.hackerrank.com
* https://www.kdnuggets.com/2019/12/most-demand-tech-skills-data-scientists.html
* https://medium.com/@jeffhale/the-most-in-demand-skills-for-data-scientists-4a4a8db896db?source=email-8029e04d384e-1571997772292-digest.weekly------0-59------------------53441023_307a_4474_bebf_2fd09612ffeb-1-----&sectionName=top
* https://www.kdnuggets.com/2019/10/growing-need-skills-data-science.html
* https://www.kdnuggets.com/2019/09/core-hot-data-science-skills.html
* https://www.kdnuggets.com/2020/01/psu-fast-track-data-science-career.html
* https://www.kdnuggets.com/2019/12/4-ways-not-hired-data-scientist.html

Trends
==========

https://www.kdnuggets.com/2020/01/top-10-technology-trends-2020.html

Strategy
==========

* [Data Sc Interview Resources](https://t.co/8JQ3MWsjks?amp=1) conordewey.com

* Prepare for job interviews

	What experience do you have in real projects?Which SQL query would
    you write to extract this information from that database? Do you
    know Docker and Kubernetes? What about Spark? Have you administered
    a Hadoop cluster? Have you used Elastic? What experience do you
    have with Kafka? OK… there are countless technologies we haven’t
    touched (so far) which may come up at a certain point. But I
    consider them as add-ons you’ll need (or not), with the only
    objective of passing the interview for a position where additional
    knowledge is needed (or not xD). Don’t think too much about this,
    and never use it as an excuse to postpone your first interview:
    you’ve already learned a lot of things, which by the way were far
    more important and complicated. As a tip: If you see that some
    requirement is repeated a lot in job offers that you like… maybe
    you should keep an eye on it. If you go to an interview and fail in
    a question… take note, go home, and strengthen your knowledge on
    that subject. Doing interviews is part of the journey; the most
    important thing is that you must assimilate it from the beginning
    and learn from it!

Learn by doing: The best way to learn something is to put it into practice. Spend most of your time writing code. And I’ll say it again: do projects!

Organize your agenda. Try to spend some time learning each day. Set small milestones with realistic deadlines (you can use a board if that helps you) and try to meet them. Check what you could accomplish and what you couldn’t. Don’t get overwhelmed, but do not relax either :)

Learn as if you had to teach: Take notes, make summaries, draw diagrams… very good! But you don’t really understand something unless you can explain it to your grandmother. And that’s the reason why I decided to start this blog :) You can follow the Feynman technique.

Take the top-down approach. With a bottom-up approach, what we would do is to follow the classic flow of learning: learn first all the small pieces before you can reach the whole. An example of this approach would be to choose an Algebra course, another one for Calculus and one more for Probability and Statistics, with the only purpose of being able to face the Machine Learning algorithms. With a top-down approach, we’ll simply try to learn Machine Learning, scratching (or deepening) the mathematical part when necessary.This way we won’t lose the motivation, the focus, or our time with something maybe irrelevant. Did you learn what offsides is before playing your first soccer game?

Be resourceful: there are a lot of available resources and tools (click on the link!). As important as having a solid knowledge is the ability to quickly locate what we don’t know or don’t remember.

Upload your projects to GitHub. Those will be your credentials to apply for the job you want. If you don’t have paid experience, you’ll need to prove experience with your own projects. On the Internet, you can find a lot of ideas or papers, and you can also try to solve a real problem or a concern of your day-to-day.

Don’t let yourself be drownedby the amount of information published daily. There are many people doing very interesting and innovative things, but you need to focus on acquiring the base that will enable you to become a data scientist; you can ignore what is published every minute on Twitter.

Prepare the interviews thoroughly. There’s a lot of information on this (lists of typical questions, tips to improve your CV, …) and even mentors if you need extra help. By the way: it is essential that you are able to explain what you did in your data projects.

Remain up to date once the goal is reached. Subscribe to the most relevant blogs and newsletters, follow the data gurus on Twitter or Linkedin, participate in forums, attend meetups, try to be Gold in a Kaggle competition, or simply expand your skills.

Rerefences
===========

* https://www.kdnuggets.com/2020/01/wanna-be-data-scientist.html
* https://www.kdnuggets.com/2020/01/top-5-data-science-skills-2020.html
* https://www.kdnuggets.com/2020/01/data-scientist-job-dream-company.html
* https://towardsdatascience.com/why-youre-not-a-job-ready-data-scientist-yet-1a0d73f15012
* https://www.kdnuggets.com/2020/08/employers-expecting-data-scientist-role-2020.html


From job in Landing AI
-----------------------

Senior Data Scientist
LATAM /
Landing AI – Engineering /
Full time
Landing AI, an Artificial Intelligence company founded by Andrew Ng, will help enterprises transform for the age of AI. This is a chance for you to get in on the ground floor of an exciting AI company. We are looking for a Senior Data Scientist who will work with our enterprise customers along with our engineer and business team to generate insights from analyzing industrial process related data. We expect you to have strong experience using a variety of data mining/data analysis methods, using a variety of data tools, building and implementing models, using/creating algorithms and creating/running simulations. The right candidate will have a passion for discovering solutions hidden in large data sets and working collaboratively to improve business outcomes.
Here’s what you will do:

        Work with customers to identify opportunities by asking the right questions to begin the discovery process 
        Acquire, process, integrate and clean data through scalable methods and develop automation frameworks for the same that can be productized
        Data investigation and exploratory analysis to understand customer applications to develop right algorithms and success criteria and testing methodologies
        Apply DS techniques, such as statistical modeling, machine learning and AI.
        Ability to understand deep data relationships using date network / connection techniques using graph, no-SQL and SQL databases
        Build testing frameworks and methodologies to measure and improve performance results of various algorithms and ML models
        Ability to use all the above to solve problems for customers by measuring results, making adjustments with feedback 

Here’s the background we’d like you to have:

        Minimum 5+ years of relevant experience with projects that are launched into production.
        Strong development experience using statistical computer languages (Python, Java, R, SQL, etc.) to manipulate data and draw insights from large data sets.
        Experience in identifying data patterns with a sense of pattern detection and anomaly detection delivering sophisticated results 
        Knowledge of a variety of machine learning techniques (such as clustering, decision trees, neural networks, etc.) and their real-world advantages/drawbacks.
        Knowledge of advanced statistical techniques and concepts (regression, properties of distributions, statistical tests and proper usage, etc.) and experience with applications.
        Experience with writing programs analyze large complex datasets to uncover complex problems
        Curiosity to dive deep into the problems to look beyond on the surface problems to discover patterns and solutions within the data.
        Excellent written and verbal communication skills for coordinating across teams.
        Drive to learn and master new technologies and techniques.
        Strong analytical and problem solving skills.
        An ability to serve as tech lead and guide the work of more junior scientists/engineers Experience in working with any graph databases is a plus.
        Experience with data modeling tools and techniques is a plus.

To be successful at Landing AI, you will have to fit well with our Landing AI Principles. Please take the time to read and understand them as they define who we are and what we look for in our candidates.

Although we are headquartered in Palo Alto, we are open to remote workers as long as they are within a 4 to 5 time zone difference to California (PDT / GMT -7).
