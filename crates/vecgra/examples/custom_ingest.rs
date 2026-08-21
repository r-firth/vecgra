use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use vecgra::{Database, DatabaseOptions, Result, Value};

struct CustomerRecord {
    external_id: &'static str,
    name: &'static str,
    active: bool,
    embedding: [f32; 4],
}

struct ProductRecord {
    external_id: &'static str,
    name: &'static str,
    sku: &'static str,
    price: f64,
    embedding: [f32; 4],
}

struct PurchaseRecord {
    customer_id: &'static str,
    product_id: &'static str,
    order_id: &'static str,
    quantity: i64,
    embedding: [f32; 4],
}

fn text(value: &str) -> Value {
    Value::String(Arc::from(value))
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("customer-orders.vg"));

    // These records can come from a CSV reader, an API, or an application.
    let customers = [CustomerRecord {
        external_id: "customer:ada",
        name: "Ada Lovelace",
        active: true,
        embedding: [1.0, 0.2, 0.0, 0.0],
    }];
    let products = [
        ProductRecord {
            external_id: "product:keyboard",
            name: "Mechanical keyboard",
            sku: "KB-01",
            price: 129.0,
            embedding: [0.8, 0.4, 0.0, 0.0],
        },
        ProductRecord {
            external_id: "product:mouse",
            name: "Wireless mouse",
            sku: "MS-02",
            price: 59.0,
            embedding: [0.7, 0.5, 0.0, 0.0],
        },
    ];
    let purchases = [
        PurchaseRecord {
            customer_id: "customer:ada",
            product_id: "product:keyboard",
            order_id: "order-1001",
            quantity: 1,
            embedding: [0.9, 0.3, 0.0, 0.0],
        },
        PurchaseRecord {
            customer_id: "customer:ada",
            product_id: "product:mouse",
            order_id: "order-1001",
            quantity: 2,
            embedding: [0.85, 0.35, 0.0, 0.0],
        },
    ];

    let database = Database::create(&path, DatabaseOptions::new(4))?;
    let mut transaction = database.transaction();
    let mut node_ids = HashMap::new();

    for record in customers {
        let id = transaction.create_node(
            "Customer",
            [
                ("name", text(record.name)),
                ("active", Value::Bool(record.active)),
            ],
            &[record.embedding.to_vec()],
        );
        node_ids.insert(record.external_id, id);
    }
    for record in products {
        let id = transaction.create_node(
            "Product",
            [
                ("name", text(record.name)),
                ("sku", text(record.sku)),
                ("price", Value::Float(record.price)),
            ],
            &[record.embedding.to_vec()],
        );
        node_ids.insert(record.external_id, id);
    }
    for record in purchases {
        transaction.create_edge(
            node_ids[record.customer_id],
            node_ids[record.product_id],
            "PURCHASED",
            [
                ("order_id", text(record.order_id)),
                ("quantity", Value::Int(record.quantity)),
            ],
            &[record.embedding.to_vec()],
        );
    }
    transaction.commit()?;

    let stats = database.read().stats();
    println!("database\t{}", path.display());
    println!("nodes\t{}", stats.nodes);
    println!("edges\t{}", stats.edges);
    println!("vectors\t{}", stats.indexed_vectors);
    Ok(())
}
