from pathlib import Path

from parser.vision import detect_text, Observation, BoundingBox


def test_parse_image():
    path = Path(__file__).parent / "food.jpg"
    annotations = detect_text(str(path))
    assert annotations == [
        Observation(
            "HMS",
            0.5,
            BoundingBox.from_bbox(
                0.3307692375954885,
                0.8517441862858737,
                0.3115384447705615,
                0.039244185495113104,
            ),
        ),
        Observation(
            "HOST",
            0.5,
            BoundingBox.from_bbox(
                0.41913838646242063,
                0.8384877450803978,
                0.19249246059319913,
                0.014884975075063034,
            ),
        ),
        Observation(
            "Feeling Good on the Move®",
            1.0,
            BoundingBox.from_bbox(
                0.36538461658034993,
                0.824127906911569,
                0.31923077482006923,
                0.011627907252443448,
            ),
        ),
        Observation(
            "An Autogrill Company",
            1.0,
            BoundingBox.from_bbox(
                0.3807692334134614,
                0.7965116278054643,
                0.2115384615384615,
                0.011627907252443448,
            ),
        ),
        Observation(
            "HMSHost Services India Pvt Ltd",
            0.5,
            BoundingBox.from_bbox(
                0.17950542639044637,
                0.754445823805612,
                0.606210673685039,
                0.02383436550751583,
            ),
        ),
        Observation(
            "Jones the Grocer Dom T1",
            1.0,
            BoundingBox.from_bbox(
                0.23735338602640316,
                0.7360510519827901,
                0.46371146960136217,
                0.02150028845223273,
            ),
        ),
        Observation(
            "Kempegowda International Airport,",
            1.0,
            BoundingBox.from_bbox(
                0.14004607321435236,
                0.7117785825769366,
                0.6662340548448947,
                0.034131540119318626,
            ),
        ),
        Observation(
            "Devanahalli, Karnataka, Bengaluru",
            0.5,
            BoundingBox.from_bbox(
                0.1401790061432768,
                0.6924078852478568,
                0.6651171450213198,
                0.03032644545834362,
            ),
        ),
        Observation(
            "THIS IS A TAX INVOICE",
            1.0,
            BoundingBox.from_bbox(
                0.25993421011025464,
                0.6598880160713487,
                0.42231996417482254,
                0.023528475787758296,
            ),
        ),
        Observation(
            "910040267 MANU",
            1.0,
            BoundingBox.from_bbox(
                0.0780542043315248,
                0.6460837721853339,
                0.2900505485115471,
                0.026167930160438058,
            ),
        ),
        Observation(
            "WS#: 100013",
            1.0,
            BoundingBox.from_bbox(
                0.6537852670569793,
                0.6219676743050289,
                0.23089100764347958,
                0.0162390701019961,
            ),
        ),
        Observation(
            "CHK 355995",
            1.0,
            BoundingBox.from_bbox(
                0.37823781745338664,
                0.5881868695508969,
                0.20507984370975701,
                0.02139714936525139,
            ),
        ),
        Observation(
            "25 Apr'24 19:31 PM",
            1.0,
            BoundingBox.from_bbox(
                0.3017444385771662,
                0.5689700159480593,
                0.36214514093084654,
                0.025281998334010014,
            ),
        ),
        Observation(
            "Take-Out",
            1.0,
            BoundingBox.from_bbox(
                0.3276094236969019,
                0.5323433115441211,
                0.31337112035506814,
                0.025249151893742217,
            ),
        ),
        Observation(
            "1 Croissant Avocado",
            1.0,
            BoundingBox.from_bbox(
                0.10454661698360772,
                0.5179875373875069,
                0.3881107372242016,
                0.03280518726749304,
            ),
        ),
        Observation(
            "Credit Card",
            1.0,
            BoundingBox.from_bbox(
                0.1471939058875942,
                0.5066624356300303,
                0.23230097844050482,
                0.02322630329026698,
            ),
        ),
        Observation(
            "19:31 B by: MANU",
            1.0,
            BoundingBox.from_bbox(
                0.14733800565364594,
                0.4850183230904914,
                0.32840091579563013,
                0.02560288866580529,
            ),
        ),
        Observation(
            "449.00",
            1.0,
            BoundingBox.from_bbox(
                0.6422211242179501,
                0.5115410639458875,
                0.12325006002908223,
                0.017615546179081232,
            ),
        ),
        Observation(
            "INR471.50",
            0.5,
            BoundingBox.from_bbox(
                0.5844917754531889,
                0.4940136364547799,
                0.1848626049446973,
                0.019240168576741046,
            ),
        ),
        Observation(
            "Subtotal",
            1.0,
            BoundingBox.from_bbox(
                0.14593144373439149,
                0.4531850167922675,
                0.16582941746973734,
                0.018048570959607546,
            ),
        ),
        Observation(
            "CGST 2,5%",
            0.5,
            BoundingBox.from_bbox(
                0.1461538457180596,
                0.4316298341119217,
                0.1884615412561885,
                0.0204050494821032,
            ),
        ),
        Observation(
            "SGST 2,5%",
            0.5,
            BoundingBox.from_bbox(
                0.14230769554122313,
                0.4126381215259407,
                0.1884615272829384,
                0.020501412739411307,
            ),
        ),
        Observation(
            "Rounding",
            1.0,
            BoundingBox.from_bbox(
                0.14230769982006922,
                0.39534883732539283,
                0.16923075805216922,
                0.018895348132644596,
            ),
        ),
        Observation(
            "Payment",
            1.0,
            BoundingBox.from_bbox(
                0.14208971963528197,
                0.37907372333687583,
                0.14658978919843177,
                0.016561855927356706,
            ),
        ),
        Observation(
            "Change Due",
            1.0,
            BoundingBox.from_bbox(
                0.08260983356200334,
                0.3528578586254517,
                0.4038429958916409,
                0.025667353888242928,
            ),
        ),
        Observation(
            "INR449.00",
            0.5,
            BoundingBox.from_bbox(
                0.5923076927949826,
                0.4418604655412358,
                0.18846154125618853,
                0.018895348132644596,
            ),
        ),
        Observation(
            "INR11.23",
            0.5,
            BoundingBox.from_bbox(
                0.6076923142431462,
                0.4229651163951603,
                0.16923075805216925,
                0.018895348132644485,
            ),
        ),
        Observation(
            "INR 11.23",
            0.5,
            BoundingBox.from_bbox(
                0.607692312691975,
                0.4040055249945741,
                0.16538460700066537,
                0.01899171270718225,
            ),
        ),
        Observation(
            "INRO.04",
            0.5,
            BoundingBox.from_bbox(
                0.6269230817304365,
                0.38517441876798,
                0.1461538377698962,
                0.018895348132644596,
            ),
        ),
        Observation(
            "INR471.50",
            0.5,
            BoundingBox.from_bbox(
                0.5844320803124814,
                0.3660361114783496,
                0.18882814344469,
                0.020834754185123372,
            ),
        ),
        Observation(
            "INRO.00",
            0.5,
            BoundingBox.from_bbox(
                0.6115384680251974,
                0.3430232562207076,
                0.28461537343678456,
                0.02470930110025138,
            ),
        ),
        Observation(
            "-- Check Closed",
            1.0,
            BoundingBox.from_bbox(
                0.302232141908629,
                0.31261311231212663,
                0.3266341939513937,
                0.025020359629425504,
            ),
        ),
        Observation(
            "25 Apr'24 19:31 PM",
            1.0,
            BoundingBox.from_bbox(
                0.30182413349041953,
                0.29277596544709816,
                0.3738578265403217,
                0.02808652530058975,
            ),
        ),
        Observation(
            "Thank you,",
            1.0,
            BoundingBox.from_bbox(
                0.3764659023204276,
                0.23161973220577503,
                0.21730600433908537,
                0.027948895870651302,
            ),
        ),
        Observation(
            "Please Visit Us Again",
            1.0,
            BoundingBox.from_bbox(
                0.255022612373212,
                0.21119940579378182,
                0.44028847383492153,
                0.0312574001965602,
            ),
        ),
        Observation(
            "GSTIN: 29AABCH7805C1ZR",
            0.5,
            BoundingBox.from_bbox(
                0.25497050232362556,
                0.13418468759677415,
                0.4591251415210766,
                0.028600355538215383,
            ),
        ),
        Observation(
            "SAC/HSN code: 996331",
            0.5,
            BoundingBox.from_bbox(
                0.27445609222993433,
                0.11637521498698722,
                0.4166520227006067,
                0.027056375261169774,
            ),
        ),
        Observation(
            "FSSAI License ID : 10023808000024",
            1.0,
            BoundingBox.from_bbox(
                0.12851348274870955,
                0.09377121271923561,
                0.6852529896047963,
                0.03467952612355274,
            ),
        ),
        Observation(
            "Feedback & Suggestions",
            1.0,
            BoundingBox.from_bbox(
                0.2519214028400639,
                0.07940690764553093,
                0.4581144283979367,
                0.02807382720610052,
            ),
        ),
        Observation(
            "E-mail :",
            0.5,
            BoundingBox.from_bbox(
                0.3948231139340608,
                0.06882312843672511,
                0.16857362404847753,
                0.015975230965166398,
            ),
        ),
        Observation(
            "kathhiravan.radhakrishnan@hmshost.net",
            0.5,
            BoundingBox.from_bbox(
                0.09045536131175669,
                0.04448680308859443,
                0.7500283377511161,
                0.03138365929956599,
            ),
        ),
        Observation(
            "Phone: +91-6366911201",
            1.0,
            BoundingBox.from_bbox(
                0.2517768532482669,
                0.03298624091474378,
                0.4460477444715116,
                0.024151622919746485,
            ),
        ),
    ]
