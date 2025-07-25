from config import collection

collection.insert_many([
    {
        "customer_id" : "C001",
        "age" : 30,
        "tenure" : 3,
        "services" : ["Internet","Phone"],
        "monthly_charge" : 65.0
    },
    {
        "customer_id" : "C002",
        "age" : 45,
        "tenure" : 6,
        "services" : ["Internet"],
        "monthly_charge" : 55.0
    },
    {
        "customer_id" : "C003",
        "age" : 28,
        "tenure" : 1,
        "services" : ["Phone","TV","Internet"],
        "monthly_charge" : 80.0
    }
])
print("Data Saved")