from glob import glob
import os

def get_file_paths(input_path):
    print(input_path)
    queries_fname = glob(os.path.join(input_path, "*queries*"))[0]
    candidates_fname = glob(os.path.join(input_path, "*candidates*"))[0]
    return queries_fname, candidates_fname

def save_ta2_output(output_dir, run_id, scores, query_labels, candidate_labels, queries_fname):

    file_prefix = os.path.basename(queries_fname).split(f"_TA2_input_queries.jsonl")[0]

    np.save(
        os.path.join(output_dir, file_prefix + f"_TA2_query_candidate_attribution_scores_{run_id}.npy"),
        scores
    )

    fout = open(
        os.path.join(
            output_dir,
            file_prefix +
            f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt"
        ), "w+"
    )
    if query_labels[0][0] == "(":
        tuple_str = True
    else:
        tuple_str = False
    if not tuple_str:
        for label in query_labels:
            fout.write("('"+str(label)+"',)")
            fout.write("\n")
        fout.close()
    else:
        for label in query_labels:
            fout.write(label)
            fout.write("\n")
        fout.close()

    fout = open(
        os.path.join(
            output_dir,
            file_prefix +
            f"_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt"
        ), "w+"
    )

    if not tuple_str:
        for label in candidate_labels:
            fout.write("('"+str(label)+"',)")
            fout.write("\n")
        fout.close()
    else:
        for label in candidate_labels:
            fout.write(label)
            fout.write("\n")
        fout.close()