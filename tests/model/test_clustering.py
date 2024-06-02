from model.clustering import cluster
from pathlib import Path

from parser.vision import detect_text


def test_clustering():
    path = Path(__file__).parent / "test1.png"
    results = detect_text(str(path))

    all_text = [observation.text for observation in cluster(path, results)]
    assert all_text == [
        "BLR LS HARVEST MARKET",
        "TAX INVOICE",
        "HMSHost Services India Pvt Ltd,",
        "KEMPEGOWDA INTERNATIONAL AIRPORT,",
        "DEVANHALLI, BENGALURU,",
        "KARNATAKA-560300",
        "POS GSTN NO : 29AABCH7805C1ZR",
        "Order No ÷11297",
        "BILL NO : 58800011293",
        "DATE AND TIME PAX",
        "21/05/2019 8:41 PM",
        "CASHIER :VINOD KUMAR MU-BLR",
        "Qty Menu Item Amount",
        "SAC/HSN Code: 996331",
        "2 PASTA 850.00",
        "2 GRILLED CHICKEN 150.00",
        "PANINI GVC 275.00",
        "2 AEREATED DRINKS (MRP) 114.30",
        "1 PAPERBOAT 120MRP 114.29",
        "Total 1503.59",
        "OGST 2.5% 37.59",
        "SGST 2.5% 37.59",
        "Round Off 0.23",
        "Grand Total 1579.00",
        "QUESTIONS & COMMENTS-",
        "Duty Manager :- +91-8884450003",
        "kathhiravan.radhakrishnanehmshost.net",
        "Credit Card MASTERS",
    ]
