# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statistics
import nltk 


from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn


pd.options.mode.chained_assignment = None
nltk.download('all')


# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics.pairwise import linear_kernel


# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat



# Recommendation Sys
@st.cache
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
	# Index of course
	idx = course_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[1] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲URL:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

# Search For Course 
@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['course_title'].str.contains(term)]
	return result_df



#COLORS for Plotly
color_discrete_map= {'Pay': 'rgb(11, 161, 123)',
                     'Free': 'rgb(231, 223, 198)',
                     'Web Development': 'rgb(18, 151, 147)',
                     'Business Finance'  : 'rgb(4, 37, 36)',
                     'Musical Instruments': 'rgb(94, 116, 115)',
                     'Graphic Design':'rgb(173, 184, 183)',
                     'All Levels':'rgb(76, 84, 84)',
                     'Beginner Level':'rgb(255, 113, 91)',
                     'Intermediate Level':'rgb(246, 248, 221)',
                     'Expert Level':'rgb(30, 168, 150)'}
#COLORS for Seaborn
palette={"All Levels":"#4c5454",
         "Beginner Level":"#ff845b",
         "Intermediate Level":"#f6f8dd",
         "Expert Level":"#1ea896",
         "Pay":"#0ba17b",
         "Free":"#e7dfc6",
         "Web Development":"#129793",
         "Business Finance":"#042524",
         "Musical Instruments":"#5e7473",
         "Graphic Design":"#adb8b7"} 

def main():

	st.title("Course Recommendation App")

	menu = ["Home","Recommend","About","EDA"]
	choice = st.sidebar.selectbox("Menu",menu)

	df = load_data("Data/udemy_courses.csv")

	if choice == "Home":
		st.subheader("Home")
		st.dataframe(df.head(10))


	elif choice == "EDA":
		st.write("## Overview of the Dataset")
		st.write(df.head())

		st.write("## Number of subjects/categories in the dataset")
		st.write(len(df['subject'].unique()))

		df.replace(to_replace= True, value = "Pay", inplace=True )
		df.replace(to_replace= False, value = "Free", inplace=True )

		## Checking for null values
		st.write("Null values in each column:")
		st.write(df.isnull().sum())

		## Displaying unique values in each column
		st.write("Unique values in each column:")
		st.write(df.nunique())

		## Removing duplicate rows
		df = df.drop_duplicates()

		## Displays information of updated dataframe
		st.write(df.info())

		## Length of free courses
		st.write("Length of free courses: ", len(df.loc[(df.is_paid=='Free')]))

		## Comparing number of free courses and courses with price = 0
		st.write("Number of free courses and courses with price = 0 are equal: ", len(df.loc[(df['is_paid'] == 'Free') & (df['price'] == 0)]) == len(df.loc[(df.is_paid=='Free')]))

		st.write("## Distribution of subjects/categories in the dataset")
		subject_counts = df['subject'].value_counts()
		st.write(subject_counts)

		st.write("## Plot of the distribution of subjects/categories")
		st.bar_chart(subject_counts)

		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.write("## Pie chart of the distribution of subjects/categories")
		plt.figure(figsize=(10,5))
		subject_counts.plot(kind='pie')
		st.pyplot()

		st.write("## Number of subscribers per subject")
		subscribers_per_subject = df.groupby('subject')['num_subscribers'].sum()
		st.write(subscribers_per_subject)

		st.write("## Plot of the number of subscribers per subject")
		st.bar_chart(subscribers_per_subject)

		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.write("## Pie chart of the number of subscribers per subject")
		plt.figure(figsize=(10,5))
		subscribers_per_subject.plot(kind='pie')
		st.pyplot()

		st.write("## Total Number of Subscribers:", df['num_subscribers'].sum())
		st.write("## Average number of subscribers:", df['num_subscribers'].mean())
		st.write("## Min number of subscribers:", df['num_subscribers'].min())
		st.write("## Max number of subscribers:", df['num_subscribers'].max())

		highest_sub = df.loc[df['num_subscribers'].idxmax()]['course_title']
		st.write("## The course with the highest number of subscribers:", highest_sub)

		# Distribution of Courses per Level
		st.set_option('deprecation.showPyplotGlobalUse', False)
		level_distribution = df['level'].value_counts()
		st.write("## Distribution of Courses per Level")
		st.write(level_distribution)

		# Distribution of Courses per Level - Plot
		st.set_option('deprecation.showPyplotGlobalUse', False)
		level_distribution_plot = level_distribution.plot(kind='bar')
		st.write("## Distribution of Courses per Level - Plot")
		st.pyplot()

		# Distribution of Subscribers per Level - Plot
		st.set_option('deprecation.showPyplotGlobalUse', False)
		subscribers_per_level = df.groupby('level')['num_subscribers'].sum()
		st.write("## Distribution of Subscribers per Level - Plot")
		subscribers_per_level_plot = subscribers_per_level.plot(kind='bar')
		st.pyplot()

		# Distribution of Levels per Subject Category
		st.set_option('deprecation.showPyplotGlobalUse', False)
		levels_per_subject = df.groupby('subject')['level'].value_counts()
		st.write("## Distribution of Levels per Subject Category")
		st.write(levels_per_subject)

		# Distribution of Levels per Subject Category - Plot
		st.set_option('deprecation.showPyplotGlobalUse', False)
		levels_per_subject_plot = levels_per_subject.plot(kind='bar')
		st.write("## Distribution of Levels per Subject Category - Plot")
		st.pyplot()

		# Seaborn Plot for `num_subscribers`
		st.set_option('deprecation.showPyplotGlobalUse', False)
		plt.figure(figsize=(20,10))
		sns.barplot(x='level',y='num_subscribers', hue='subject',data=df)
		st.write("## Seaborn Plot for `num_subscribers`")
		st.pyplot()
		
		# Seaborn Plot for `num_lectures`
		st.set_option('deprecation.showPyplotGlobalUse', False)
		plt.figure(figsize=(20,10))
		sns.barplot(x='level',y='num_lectures', hue='subject',data=df,ci=None)
		st.subheader("Seaborn Plot for `num_lectures`")
		st.pyplot()

		# Function to Plot the Distribution of `countable_feature`
		def plot_num_of_countable_feature(countable_feature):
			# Seaborn Plot
			plt.figure(figsize=(10,7))
			plt.title("Plot of {} per level per subject".format(countable_feature))
			sns.barplot(x='level',y=countable_feature, hue='subject',data=df,ci=None)
			st.write("Seaborn Plot for `{}`".format(countable_feature))
			st.pyplot()

			# Pie Plot
			plt.figure(figsize=(10,7))
			plt.title("Plot of {} per level".format(countable_feature))
			df.groupby('level')[countable_feature].sum().plot(kind='pie')
			st.write("Pie Plot for `{}`".format(countable_feature))
			st.pyplot()

		# Datatype
			st.write("Datatype of the `price` column: ", df['price'].dtype)
			st.write("Unique prices in the dataset: ", df['price'].unique())
			st.write("The average price of courses: ", df['price'].mean())
			st.write("The maximum price of courses: ", df['price'].max())

			# Most Profitable Course
			st.write("Computing the profit for each course (price x num_subscriber)...")
			df['profit'] = df['price'] * df['num_subscribers']
			st.write("The most profitable course: ", df['profit'].max())

		# Show scatter plot of price vs num_subscribers
		st.title("Scatter Plot of Price vs Number of Subscribers")

		chart = alt.Chart(df).mark_circle().encode(
			x='price',
			y='num_subscribers'
		)

		st.altair_chart(chart, use_container_width=True)
		# Show scatter plot of price vs num_reviews
		st.title("Scatter Plot of Price vs Number of Reviews")

		chart = alt.Chart(df).mark_circle().encode(
			x='price',
			y='num_reviews'
		)

		st.altair_chart(chart, use_container_width=True)

		df['published_timestamp']=pd.to_datetime(df['published_timestamp'])
		df['year']=df['published_timestamp'].dt.year
		df['content_duration']=(df['content_duration']*60).astype(int)

		st.subheader("Percentage of Free Courses & Paid Courses")
		is_paid = df.groupby(['is_paid']).count()[['course_id']].reset_index()
		st.write(is_paid)
		pie1 = px.pie(is_paid, values='course_id', names='is_paid', color='is_paid', color_discrete_map=color_discrete_map)
		pie1.update_traces(textposition='inside', textinfo='percent+label')
		pie1.update_layout(showlegend=False)
		st.plotly_chart(pie1)

		st.subheader("Percentage of Courses per Level")
		level = df.groupby(['level']).count()[['course_id']].reset_index()
		st.write(level)
		pie3 = px.pie(level, values='course_id', names='level', color='level', color_discrete_map=color_discrete_map)
		pie3.update_traces(textposition='outside', textinfo='percent+label')
		pie3.update_layout(showlegend=False)
		st.plotly_chart(pie3)

		st.subheader("Percentage of Courses per Subject")
		subject = df.groupby(['subject']).count()[['course_id']].reset_index()
		st.write(subject)
		pie2 = px.pie(subject, values='course_id', names='subject', color='subject', color_discrete_map=color_discrete_map)
		pie2.update_traces(textposition='inside', textinfo='percent+label')
		pie2.update_layout(showlegend=False)
		st.plotly_chart(pie2)

		st.subheader("Percentage of Subscribers per Subject")
		sub_perc = df.groupby(['subject']).sum()[['num_subscribers']].sort_values(['num_subscribers'],ascending=False).reset_index()
		sub_perc['perc'] = round(sub_perc.num_subscribers/sum(sub_perc.num_subscribers)*100, 2)
		sub_perc['cum_perc'] = round(sub_perc['num_subscribers'].cumsum()/sub_perc['num_subscribers'].sum()*100, 2)
		st.write(sub_perc.reset_index().to_markdown())


		st.write("Conclusions")
		st.write("1.The most courses are paid 91.6 percent of them).")
		st.write("2.The courses that have been posted the number of courses for all levels is almost equal to the rest of the levels.")
		st.write("3.Τhe subjects with the most courses are in Web Development and Business Finance and follows Musical Instruments and Graphic Design.")

		# Scatter plot of Subscribers and Reviews
		st.subheader("Marginal Distribution Plots of Reviews & Subscribers ")
		sub_rev_sc = px.scatter(df, 
								x="num_subscribers", 
								y="num_reviews",
								labels={"num_reviews":"Reviews","num_subscribers":"Subscribers"},
								marginal_x="histogram", 
								marginal_y="rug", 
								trendline="ols", 
								trendline_color_override="red")

		sub_rev_sc.update_layout(title_text='Marginal Distribution Plots Reviews & Subscribers ',title_font=dict(size=18))
		st.plotly_chart(sub_rev_sc)

		# Scatter plot of Content Duration and Number of Lectures
		st.subheader("Marginal Distribution Plots of Content Duration & Number of Lectures ")
		dur_lec_sc = px.scatter(df, 
								x="num_lectures", 
								y="content_duration",
								labels={"content_duration":"Content Duration","num_lectures":"Number of Lectures"},
								marginal_x="histogram", 
								marginal_y="rug", 
								trendline="ols", 
								trendline_color_override="red")

		dur_lec_sc.update_layout(title_text='Marginal Distribution Plots Content Duration & Number of Lectures',title_font=dict(size=18))
		st.plotly_chart(dur_lec_sc)

		st.subheader("Marginal Distribution Plots of Price & Number of Reviews ")
		price_reviews_sc = px.scatter(df, 
							x="num_reviews", 
							y="price",
							labels={"price":"Price","num_reviews":"Number of Reviews"},
							marginal_x="histogram", 
							marginal_y="rug", 
							trendline="ols", 
							trendline_color_override="red")

		price_reviews_sc.update_layout(title_text='Marginal Distribution Plots of Price & Number of Reviews ',title_font=dict(size=18))
		st.plotly_chart(price_reviews_sc)

		st.subheader("Marginal Distribution Plots of Subscribers & Prices ")
		sub_price_sc = px.scatter(df, 
								x="num_subscribers", 
								y="price",
								labels={"price":"Price","subscribers":"Subscribers"},
								marginal_x="histogram", 
								marginal_y="rug", 
								trendline="ols", 
								trendline_color_override="red")

		sub_price_sc.update_layout(title_text='Marginal Distribution Plots Subscribers & Prices ',title_font=dict(size=18))
		st.plotly_chart(sub_price_sc)

		st.subheader("Courses with most Subscribers per Subject")
		fig_top_all=px.bar(df, 
                   y="course_title", 
                   x="num_subscribers",
                   orientation='h',
                   labels={"course_title":"","num_subscribers":"Subscribers","subject":"Subject"},
                   color='subject',
                   color_discrete_map=color_discrete_map,
                   text='is_paid',
                   height=1000)
		fig_top_all.update_layout(title_text='Courses with most Subscribers per Subject')
		fig_top_all.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'})
		st.plotly_chart(fig_top_all)

		st.subheader("Boxplot of Subscribers per Subject")
		fig=px.box(df, x="subject", y="num_subscribers",color='subject',color_discrete_map=color_discrete_map,labels={"num_subscribers":"Subscribers","subject":"Subject"})
		fig.update_traces(quartilemethod="exclusive") 
		fig.update_layout(title_text='Boxplot of Subscribers per Subject')
		st.plotly_chart(fig)

		st.subheader("Price Range of most Subscribed courses per Subject")
		sns.displot(df, x="price", hue="subject", multiple="dodge",bins=8,palette=palette,height=8,aspect=1)
		plt.title("Price Range of most Subscribed courses per Subject", y=1.02);
		st.pyplot()

		st.subheader("Price Range of most Subscribed courses per Level")
		sns.displot(df, x="price", hue="level", multiple="dodge",bins=8,palette=palette,height=8,aspect=1)
		plt.title("Price Range of most Subscribed courses per Level", y=1.02);
		st.pyplot()

		st.subheader("Correaltion heatmap")
		dfCorr = pd.DataFrame(df.corr())
		dfCorr = dfCorr.iloc[:-1 , :-1]

		plt.figure(figsize=(10,10))
		sns.heatmap(dfCorr, annot=True)
		st.pyplot()
	
	
	elif choice == "Recommend":
		st.subheader("Recommend Courses")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
		search_term = st.text_input("Search")
		num_of_rec = st.sidebar.number_input("Number",4,30,7)
		if st.button("Recommend"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.beta_expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_url = row[1][2]
						rec_price = row[1][3]
						rec_num_sub = row[1][4]

						# st.write("Title",rec_title,)
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_url,rec_num_sub),height=350)
				except:
					results= "Not Found"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df[['course_title']])



				# How To Maximize Your Profits Options Trading

		




	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")
		st.text("Built by :Mohammed Jalaluddin Yazdani and Mohammed Siraj")
		


if __name__ == '__main__':
	main()