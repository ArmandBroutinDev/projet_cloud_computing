from utils.utils import DataHandler, FeatureRecipe, FeatureExtractor

def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """
    dt = DataHandler()
    dt.get_process_data()
    fr = FeatureRecipe(dt.df_res)
    fr.get_process_data(0.3)

    flist = ['listing_id','beds','pricing_weekly_factor','pricing_monthly_factor','name','type','city','neighborhood','latitude','longitude','is_rebookable','is_new_listing','is_fully_refundable','is_host_highly_rated']
    fe = FeatureExtractor(dt.df_res,flist)
    X_train,X_test,y_train,y_test = fe.get_process_data()

    return X_train,X_test,y_train,y_test