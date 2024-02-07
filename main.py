from fastapi import FastAPI
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

app = FastAPI()

# -----------------------------------------
data_all_reviews = pd.read_parquet('data_reviews.parquet')
steam_games = pd.read_parquet('steam_games.parquet')

steam_games['genres'] = steam_games['genres'].apply(lambda x: x.tolist())
steam_games['tags'] = steam_games['tags'].apply(lambda x: x.tolist())
steam_games['specs'] = steam_games['specs'].apply(lambda x: x.tolist())

data_items = pd.read_parquet('data_items.parquet')

# -----------------------------------------

def userdata(user_id):
    
    user_opins = pd.merge(data_all_reviews.drop(columns='sentiment'),steam_games.drop(columns=['genres','app_name','title','release_date','tags','specs','developer']), how='left', left_on='item_id',right_on='id').query('user_id == @user_id')

    user_paid = user_opins.price.sum()
    user_recom = (len(user_opins[user_opins.recommend==True]) / len(user_opins))*100
    user_items = len(user_opins)

    return {"Usuario": user_id,"Dinero Gastado": user_paid,"% de recomendación: ": user_recom, "Cantidad de Items":user_items}

@app.get("/User_Data/")
def userData(user_id : str):


    return userdata(user_id)

# -----------------------------------------

def UserGenre(genre):
    genre = genre.lower()
    User_gen = pd.merge(data_items, steam_games[steam_games.genres.map(lambda genr: genr == [genre])].loc[:,('id','release_date')], left_on='item_id', right_on='id')

    user = User_gen[['user_id','playtime_forever']].groupby('user_id')['playtime_forever'].sum().sort_values(ascending=False).head(1).index[0]
    registro = User_gen.query('user_id==@user')[['release_date','playtime_forever']].groupby(User_gen.query('user_id==@user')['release_date'].dt.year)['playtime_forever'].sum()
    
    diccionario_resultante = registro.to_dict()
    lista_de_diccionarios = [{"{}".format(key): value} for key, value in diccionario_resultante.items()]

    return {'Usuario con más horas jugadas':user,'Horas Jugadas':lista_de_diccionarios}


@app.get("/User_Genr/")
def UserForGenre(gener : str):

    return UserGenre(gener)

# -----------------------------------------

def best_developer_by_year(year):
    dev_gen = pd.merge(data_all_reviews, steam_games.loc[:,('id','release_date','developer')], left_on='item_id', right_on='id')[['recommend','sentiment','release_date','developer']].query('release_date.dt.year == @year and recommend==True and recommend==True and sentiment==2')['developer'].value_counts().head(3).to_dict()

    return dev_gen

@app.get("/best_dev/")
def best_developer_year(year : int):

    return best_developer_by_year(year)


# -----------------------------------------
def developer_reviews(developer):
    
    dev_dc = pd.merge(data_all_reviews, steam_games.loc[:,('id','release_date','developer')], left_on='item_id', right_on='id')[['sentiment','developer']].query('developer == @developer').groupby('developer').value_counts().to_dict()

    dev_neg = dev_dc.get((developer,0))
    dev_pos = dev_dc.get((developer,2))

    return {developer:[dev_neg,dev_pos]}

@app.get("/dev_review/")
def developer_reviews_analysis(dev_id : str):


    return developer_reviews(dev_id)

# -----------------------------------------

def porcentaje_juegos_gratis(dev):
    steam_games_modf = steam_games.query('developer==@dev').copy()
    free_games = steam_games_modf[steam_games_modf['price'] == 0]

    # Agrupa por fecha de lanzamiento y cuenta el total de juegos y juegos gratis
    grouped_data = steam_games_modf.groupby(steam_games_modf.release_date.dt.year).agg(total_items=('price', 'count'), free_items=('price', lambda x: (x == 0).sum()))

    # Calcula el porcentaje de juegos gratis
    grouped_data['percentage_free'] = (grouped_data['free_items'] / grouped_data['total_items']) * 100

    # Crea el diccionario
    result_dict = grouped_data[['total_items', 'percentage_free']].to_dict(orient='index')

    return result_dict


@app.get("/dev/")
def developer(dev_id : str):


    return porcentaje_juegos_gratis(dev_id)

#==============================================================

loaded_model = Word2Vec.load("word2vec_model.joblib")

# Carga el DataFrame modificado
loaded_df = pd.read_pickle("steam_games_processed.pkl")

cosine_similarities = cosine_similarity(list(loaded_df['vector']), list(loaded_df['vector']))


def get_top_n_recommendations(game_title, n=5):

    cosine_similarities = cosine_similarity(list(loaded_df['vector']), list(loaded_df['vector']))

    game_index = loaded_df.query('title==@game_title').index[0]
    
    sim_scores = list(enumerate(cosine_similarities[game_index]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_n_indices = [i for i, _ in sim_scores[1:n+1]]

    recommended_games = loaded_df['title'].iloc[top_n_indices].tolist()

    return recommended_games



@app.get("/Model_Recommendation/")
def developer(title : str):


    return get_top_n_recommendations(title)
