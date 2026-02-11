#![feature(coverage_attribute)]

mod dotenv;

use axum::serve::ListenerExt;
use nghe_api::constant;
use nghe_backend::{build, config, init_tracing, migration};

#[coverage(off)]
fn main() {
    dotenv::load();
    tokio_main();
}

#[tokio::main]
async fn tokio_main() {
    let config = config::Config::load().unwrap_or_else(|e| {
        eprintln!("Could not parse config: {e}");
        eprintln!();
        eprintln!("Missing something? Minimum required environment variables:");
        eprintln!("  NGHE_DATABASE__URL=postgres://...");
        eprintln!("  NGHE_DATABASE__KEY=<32-hex-chars>   # e.g. `openssl rand -hex 16`");
        eprintln!();
        eprintln!("Tip: create a `.env` file in the repo root with those values.");
        std::process::exit(2);
    });
    init_tracing(&config.log).unwrap();

    tracing::info!(server_version =% constant::SERVER_VERSION);
    tracing::info!("{config:?}");

    migration::run(&config.database.url).await;

    let listener = tokio::net::TcpListener::bind(config.server.to_socket_addr())
        .await
        .unwrap()
        .tap_io(|tcp_stream| tcp_stream.set_nodelay(true).unwrap());
    axum::serve(listener, build(config).await).await.unwrap();
}
