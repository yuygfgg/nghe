use leptos::prelude::*;
use leptos::{html, svg};
use nghe_api::common::filesystem;
use nghe_api::music_folder::add as add_music_folder;
use nghe_api::permission::add as add_permission;
use nghe_api::scan::start as start_scan;
use nghe_api::user::get as get_user;

use crate::Error;
use crate::client::Client;
use crate::components::ClientRedirect;
use crate::components::form;
use crate::flowbite;

pub const MODAL_ID: &str = "scan-folder-modal";

fn infer_name(path: &str) -> String {
    let path = path.trim_end_matches('/');
    path.rsplit('/').find(|s| !s.is_empty()).unwrap_or("Music").to_string()
}

fn Modal(client: Client, set_status: WriteSignal<Option<String>>) -> impl IntoView {
    let folder_path = RwSignal::new(String::default());
    let (path_error, set_path_error) = signal(Option::default());

    let scan_action = Action::<String, Result<(), Error>>::new_unsync(move |path: &String| {
        let client = client.clone();
        let path = path.trim().to_string();
        let folder_path = folder_path;
        async move {
            let response = client
                .json(&add_music_folder::Request {
                    name: infer_name(&path),
                    path: path.clone(),
                    ty: filesystem::Type::Local,
                    allow: false,
                })
                .await?;

            // Ensure we have a permission row for this folder, otherwise `startScan` will fail
            // while checking ownership.
            let me = client.json(&get_user::Request { id: None }).await?;
            client
                .json(&add_permission::Request {
                    user_id: Some(me.id),
                    music_folder_id: Some(response.music_folder_id),
                    permission: nghe_api::permission::Permission { owner: true, share: false },
                })
                .await?;

            client
                .json(&start_scan::Request {
                    music_folder_id: response.music_folder_id,
                    full: start_scan::Full::default(),
                })
                .await?;

            flowbite::modal::hide(MODAL_ID);
            folder_path.set(String::default());
            set_status.set(Some(format!("Started scan for {path}")));
            Ok(())
        }
    });

    html::div()
        .id(MODAL_ID)
        .tabindex("-1")
        .aria_hidden("true")
        .class(
            "hidden overflow-y-auto overflow-x-hidden fixed top-0 right-0 left-0 z-50 \
             justify-center items-center w-full md:inset-0 h-[calc(100%-1rem)] max-h-full",
        )
        .child(html::div().class("relative p-4 w-full max-w-md max-h-full").child(form::Form(
            "Scan folder",
            Some(MODAL_ID),
            move || {
                form::input::Text(
                    "folder_path",
                    "Folder path",
                    "text",
                    "path",
                    "/Users/a1/Music",
                    folder_path,
                    path_error,
                )
            },
            "Scan",
            move |_| {
                let path = folder_path().trim().to_string();
                let err = if path.is_empty() {
                    Some("Folder path could not be empty")
                } else if !path.starts_with('/') {
                    // Backend requires absolute paths for local scanning.
                    Some("Folder path must be an absolute path (start with /)")
                } else {
                    None
                };
                set_path_error(err);
                if err.is_some() {
                    return;
                }

                scan_action.dispatch(path);
            },
            scan_action,
        )))
}

pub fn ScanFolder() -> impl IntoView {
    ClientRedirect(move |client| {
        let node_ref = flowbite::init::suspense();
        let (status, set_status) = signal::<Option<String>>(None);

        html::div()
            .node_ref(node_ref)
            .class("m-4 p-4 bg-white dark:bg-gray-800 shadow-md sm:rounded-lg")
            .child((
                html::div().class("flex items-center justify-between").child((
                    html::h2()
                        .class("text-lg font-semibold text-gray-900 dark:text-white")
                        .child("Music library scan"),
                    html::button()
                        .r#type("button")
                        .attr("data-modal-target", MODAL_ID)
                        .attr("data-modal-toggle", MODAL_ID)
                        .class(
                            "flex items-center justify-center text-gray-900 whitespace-nowrap \
                             dark:text-white px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-600 \
                             focus:ring-3 focus:ring-gray-300 rounded-lg focus:outline-none \
                             dark:focus:ring-gray-600",
                        )
                        .child((
                            svg::svg()
                                .aria_hidden("true")
                                .class("h-4 w-4 mr-2")
                                .attr("fill", "currentColor")
                                .attr("viewBox", "0 0 20 20")
                                .attr("xmlns", "http://www.w3.org/2000/svg")
                                .child(svg::path().attr(
                                    "d",
                                    "M2 6a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v6a2 2 0 \
                                         0 1-2 2H4a2 2 0 0 1-2-2V6Z",
                                )),
                            "Scan folder",
                        )),
                )),
                html::div().class("mt-4 text-sm text-gray-600 dark:text-gray-300").child((
                    html::p().child(
                        "Enter an absolute local folder path. The server will walk the directory \
                         and import audio files into the music library.",
                    ),
                    move || {
                        status().map(|s| {
                            html::p().class("mt-3 text-green-700 dark:text-green-400").child(s)
                        })
                    },
                )),
                Modal(client, set_status),
            ))
    })
}
