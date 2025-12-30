import streamlit as st
import os
import traceback
import tempfile
import crew_analyzer
from lib import utils
from dotenv import load_dotenv
from datetime import datetime

# TODO: select a previous uploaded job from jobs.
# TODO: Check for failure to access/download the job url and inform user.
# TODO: Display the saved crew output files in an expandable section.

# -- initialize env etc --
if "initialized" not in st.session_state:
    with st.spinner("Initializing..."):
        load_dotenv()
        st.session_state.initialized = True
        utils.make_work_dirs()

        # load storage locations from .env into session state
        resume_storage_dir = os.getenv("RESUME_STORAGE_DIR")
        job_storage_dir = os.getenv("JOB_STORAGE_DIR")
        crew_output_storage_dir = os.getenv("CREW_OUTPUT_STORAGE_DIR")
        st.session_state.resume_storage_dir = resume_storage_dir
        st.session_state.job_storage_dir = job_storage_dir
        st.session_state.crew_output_storage_dir = crew_output_storage_dir

st.set_page_config(page_title="Resume Job Scorer", layout="wide")

st.title("Resume Job Scorer")
st.markdown("Upload your resume and provide a job URL to see how well you match!")


def display_usage_metrics(usage_metrics):
    usage_str = "AI LLM Token Usage Metrics\n"
    for key, value in usage_metrics.items():
        usage_str += f"{key}: {value}\n"
    st.markdown(
        f"<p style='font-size: 10px; white-space: pre-wrap; font-family: monospace;'>{usage_str}</p>",
        unsafe_allow_html=True,
    )


# Sidebar for inputs
with st.sidebar:
    st.markdown("### Candidate")
    # list last 3 uploaded resumes in ./work/resumes in a dropdown  sort by date newest to oldest
    if st.button("Rescan Resume Folder"):
        resume_files = utils.get_list_of_files_desc(st.session_state.resume_storage_dir)
    else:
        resume_files = utils.get_list_of_files_desc(st.session_state.resume_storage_dir)
    resume_files = ["Upload Resume (PDF)"] + resume_files[:3]
    resume_file = st.selectbox("Resume", resume_files, index=0)
    if resume_file == "Upload Resume (PDF)":
        resume_file = st.file_uploader(
            "Upload Resume (PDF or TXT)", type=["pdf", "txt"]
        )
        previous_resume = False
    else:
        resume_file = os.path.join(st.session_state.resume_storage_dir, resume_file)
        previous_resume = True
    resume_caching = st.checkbox(
        "Use cached result if available", key="resume_caching", value=True
    )

    st.markdown("### Job")
    if st.button("Rescan Job Folder"):
        previous_job_files = [
            "Previous Job Submissions"
        ] + utils.get_list_of_files_desc(st.session_state.job_storage_dir)
    else:
        previous_job_files = [
            "Previous Job Submissions"
        ] + utils.get_list_of_files_desc(st.session_state.job_storage_dir)

    previous_job_file = st.selectbox("Job", previous_job_files, index=0)
    job_url = st.text_input(
        "Job URL", placeholder="https://www.linkedin.com/jobs/view/..."
    )
    job_text = st.text_area("Job Text", placeholder="Paste job description here")

    job_caching = st.checkbox(
        "Use cached result if available", key="job_caching", value=True
    )
    analyze_button = st.button("Analyze Match")

    st.divider()
    # show crew .env config items

if analyze_button:
    if resume_file is not None and (
        job_url != ""
        or job_text != ""
        or previous_job_file != "Previous Job Submissions"
    ):
        with st.spinner("Preparing files..."):
            if previous_job_file != "Previous Job Submissions":
                job_url = os.path.join(
                    st.session_state.job_storage_dir, previous_job_file
                )
            elif job_text:
                # write to a temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", dir=st.session_state.job_storage_dir
                ) as tmp_file:
                    tmp_file.write(job_text.encode())
                    tmp_file_path = tmp_file.name
                job_url = tmp_file_path
            if not previous_resume:
                resume_file_extension = resume_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix="resume_" + resume_file.name.replace(" ", "_"),
                    suffix=f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{resume_file_extension}",
                ) as tmp_file:
                    tmp_file.write(resume_file.getvalue())
                    tmp_file_path = tmp_file.name
            else:
                tmp_file_path = resume_file
        try:
            with st.spinner("Analysing resume..."):
                resume_parsed_filename = os.path.basename(tmp_file_path)
                # Run the crew
                resume_analysis, resume_analysis_path, resume_crew_usage_metrics = (
                    crew_analyzer.resume_skill_analyzer_crew(
                        tmp_file_path, resume_caching
                    )
                )

                col_resume_analysis1, col_resume_analysis2 = st.columns([0.8, 0.2])
                with col_resume_analysis1:
                    st.success(
                        f"**Resume analysis complete!**  \n"
                        f"Analysis saved to: `{resume_analysis_path}`  \n"
                        f"Resume stored as: `{st.session_state.resume_storage_dir}/{resume_parsed_filename}.txt`"
                    )
                with col_resume_analysis2:
                    display_usage_metrics(resume_crew_usage_metrics)

            with st.spinner("Analysing job..."):
                job_analysis, job_analysis_path, job_details, job_crew_usage_metrics = (
                    crew_analyzer.job_requirements_analyzer_crew(job_url, job_caching)
                )

                col_job_analysis1, col_job_analysis2 = st.columns([0.8, 0.2])
                with col_job_analysis1:
                    st.success(
                        f"**Job analysis complete!**  \n"
                        f"Analysis saved to: `{job_analysis_path}`"
                    )
                with col_job_analysis2:
                    display_usage_metrics(job_crew_usage_metrics)

            with st.spinner("Analyzing Job vs Resume skills and Deciding..."):
                (
                    final_decision,
                    final_decision_path,
                    job_details,
                    job_crew_usage_metrics,
                ) = crew_analyzer.job_vs_resume_analyzer_crew(
                    job_analysis,
                    resume_analysis,
                    job_details,
                    job_caching,
                )

                col_final_decision1, col_final_decision2 = st.columns([0.8, 0.2])
                with col_final_decision1:
                    st.success(
                        f"**Final decision complete!**  \n"
                        f"Decision analysis saved to: `{final_decision_path}`"
                    )
                with col_final_decision2:
                    display_usage_metrics(job_crew_usage_metrics)

            # Parse the output
            try:
                result_json = crew_analyzer.result_to_json(final_decision)
            except Exception as e:
                st.error(f"Failed to parse result JSON: {e}")
                st.text(final_decision.raw)
                st.stop()

            # Display Results
            st.header("Analysis Results")

            # Job Details
            org = result_json.get("organization")
            job_summary = result_json.get("job_summary")

            url = job_details.get("job_url")
            source = job_details.get("job_source")

            st.subheader("Job Details")
            st.write(
                f"""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; margin-bottom: 15px;">
                    <h3 style="margin-top: 0; color: #333;">{org}</h3>
                    <p style="margin-bottom: 10px;"><strong>Source:</strong> {source}</p>
                    <p style="margin-bottom: 10px;"><strong>Role Summary:</strong> {job_summary}</p>
                    <a href="{url}" target="_blank" style="text-decoration: none;">
                        <div style="display: inline-block; background-color: #007bff; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;">
                            View Job Posting
                        </div>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

            decision = result_json.get("decision")
            reason = result_json.get("reason")
            st.subheader("Resume vs Job Decision")
            if decision == "Pass":
                st.success(
                    f"Candidate resume is a good fit for the Job! Reason: {reason}"
                )
            else:
                st.error(
                    f"Candidate resume is not a good fit for the Job! Reason: {reason}"
                )

            # Score Section
            if "score" in result_json:
                score_data = result_json["score"]
                final_score = score_data.get("final_score", 0)
                percentage = final_score * 100

                # Color coding
                if percentage >= 80:
                    color = "green"
                elif percentage >= 50:
                    color = "orange"  # Streamlit uses orange for warning/yellowish
                else:
                    color = "red"

                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                        <h2>Match Score</h2>
                        <h1 style="color: {color}; font-size: 72px;">{percentage:.1f}%</h1>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown("### 📊 Skills Analysis")
                # Detailed Score Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Required Skills Match",
                        f"{score_data.get('required_skill_match_score', 0) * 100:.1f}%",
                    )
                    st.text(
                        f"Matched: {score_data.get('matching_required_skills_count', 0)} / {score_data.get('total_required_skills_count', 0)}"
                    )
                with col2:
                    st.metric(
                        "Preferred Skills Match",
                        f"{score_data.get('preferred_skill_match_score', 0) * 100:.1f}%",
                    )
                    st.text(
                        f"Matched: {score_data.get('matching_preferred_skills_count', 0)} / {score_data.get('total_preferred_skills_count', 0)}"
                    )

            else:
                st.warning("Score data not found in the output.")

            st.divider()

            # Skills Breakdown
            col3, col4 = st.columns(2)

            with col3:
                if (
                    "missing_required_skills" in result_json
                    and result_json["missing_required_skills"]
                ):
                    skills_list = "\n".join(
                        [
                            f"- {skill}"
                            for skill in result_json["missing_required_skills"]
                        ]
                    )
                    st.error(f"**Missing:**\n\n{skills_list}")
                st.subheader("Required Skills")
                if (
                    "matching_required_skills" in result_json
                    and result_json["matching_required_skills"]
                ):
                    skills_list = "\n".join(
                        [
                            f"- {skill}"
                            for skill in result_json["matching_required_skills"]
                        ]
                    )
                    st.success(f"**Matching:**\n\n{skills_list}")

            with col4:
                if (
                    "missing_preferred_skills" in result_json
                    and result_json["missing_preferred_skills"]
                ):
                    skills_list = "\n".join(
                        [
                            f"- {skill}"
                            for skill in result_json["missing_preferred_skills"]
                        ]
                    )
                    st.error(f"**Missing:**\n\n{skills_list}")
                st.subheader("Preferred Skills")
                if (
                    "matching_preferred_skills" in result_json
                    and result_json["matching_preferred_skills"]
                ):
                    skills_list = "\n".join(
                        [
                            f"- {skill}"
                            for skill in result_json["matching_preferred_skills"]
                        ]
                    )
                    st.success(f"**Matching:**\n\n{skills_list}")

            st.divider()

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.code(traceback.format_exc())

    else:
        st.warning("Please upload or select a resume and provide a job URL.")
