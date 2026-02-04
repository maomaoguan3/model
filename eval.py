import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
SELECTED_FEATURES = [
    'inst_inst_code','eb_iban_num','inst_fund_channel_code','eb_card_brand',
    'eb_merchant_create_source','eb_cnt_by_card_amt','eb_biz_product_code',
    'eb_card_bank_code','eb_kyc_username_keyword_2','eb_first_bind',
    'eb_merchant_name_keyword_1','pmt_pay_channel','eb_pay_amount',
    'eb_pay_channel','eb_customer_event_type','eb_member_country',
    'eb_kyc_username_keyword_1','eb_bind_card_days','eb_kyc_finish_days',
    'eb_card_country','eb_register_days','eb_merchant_id','inst_amount',
    'eb_anonymous_pay','cmf_amount','eb_transaction_amount','eb_cnt_by_member_amt',
    'eb_three_ds_success_count','inst_communicate_status','eb_online',
    'eb_billing_agreement','eb_card_num','inst_product_code','eb_app_name',
    'eb_card_ocr','eb_green_point_used','eb_card_is_verified','eb_account_balance',
    'eb_event_time_minute','eb_merchant_name_keyword_2','inst_pay_mode',
    'eb_kyc_username_keyword_3','eb_bank_cardholder_name_keyword_1',
    'eb_data_eden_aml_switch','eb_on_card','eb_kyc_type','eb_payment_type',
    'eb_content_language','eb_kyc_source',
]

if __name__ == "__main__":
    test_df = pd.read_parquet("out/test_df.parquet")
    oot_df = pd.read_parquet("out/oot_df.parquet")

    model = lgb.Booster(model_file="out/lightgbm/best_model.txt")

    test_pred = model.predict(test_df[SELECTED_FEATURES])
    oot_pred = model.predict(oot_df[SELECTED_FEATURES])

    print(f"Test AUC: {roc_auc_score(test_df['is_fraud'], test_pred)}")
    print(f"OOT AUC: {roc_auc_score(oot_df['is_fraud'], oot_pred)}")


    # Binned fraud rate.
    test_df["pred"] = test_pred
    test_df["inst_inst_order_no"] = test_df["inst_inst_order_no"]
    test_df["pred_bin"] = pd.qcut(test_df["pred"], 10, labels=False)
    test_df.groupby("pred_bin")["is_fraud"].mean().plot(kind="barh")
    test_df.to_parquet("out/test_result_df.parquet")
    plt.savefig("out/lightgbm/lightgbm_binned_fraud_rate.png")

    # Binned fraud rate.
    oot_df["pred"] = oot_pred
    oot_df["inst_inst_order_no"] = oot_df["inst_inst_order_no"]
    oot_df["pred_bin"] = pd.qcut(oot_df["pred"], 10, labels=False)
    oot_df.groupby("pred_bin")["is_fraud"].mean().plot(kind="barh")
    oot_df.to_parquet("out/oot_result_df.parquet")
    plt.savefig("out/lightgbm/lightgbm_binned_fraud_rate_oot.png")