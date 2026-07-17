from tqdm.auto import tqdm
from wbfm.utils.general.utils_hardcoded import load_all_paper_datasets
from wbfm.utils.projects.finished_project_data import rename_manual_ids_from_excel_in_project

if __name__ == '__main__':
    # Load ALL of the paper projects and update the neuron names
    # This is a one-time operation, and will take a while
    all_projects_dict = load_all_paper_datasets()
    all_errors = []
    for project_name, project in tqdm(all_projects_dict.items()):
        print(project.project_dir)
        try:
            rename_manual_ids_from_excel_in_project(project)
        except PermissionError as e:
            all_errors.append(e)
        except ValueError as e:
            # This one is unusual, and should be investigated
            print(project.df_manual_ids.columns)
            raise e
    print(f"All errors: {all_errors}")
