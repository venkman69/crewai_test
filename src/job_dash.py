import streamlit as st
import os
import tempfile
import json
import resume_job_scorer
from lib import utils
from dotenv import load_dotenv
from datetime import datetime

# TODO: select a previous uploaded job from jobs.
# TODO: Move missing skills above the matching skills.
# TODO: Check for failure to access/download the job url and inform user.
# TODO: Display the saved crew output files in an expandable section.


if os.getenv("GEMINI_API_KEY") is None:
    load_dotenv()
    if os.getenv("GEMINI_API_KEY") is None:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    else:
        print("GEMINI_API_KEY found in environment variables")

st.set_page_config(page_title="Resume Job Scorer", layout="wide")

st.title("Resume Job Scorer")
st.markdown("Upload your resume and provide a job URL to see how well you match!")

# Sidebar for inputs
with st.sidebar:
    st.header("Inputs")
    # list last 3 uploaded resumes in ./parsed_files in a dropdown  sort by date newest to oldest
    st.markdown("### Last 3 Uploaded Resumes")
    resume_files = os.listdir("./parsed_files")
    resume_files.sort(
        key=lambda x: os.path.getmtime(os.path.join("./parsed_files", x)), reverse=True
    )
    resume_files = ["Upload Resume (PDF)"] + resume_files[:3]
    resume_file = st.selectbox("Resume", resume_files, index=0)
    if resume_file == "Upload Resume (PDF)":
        resume_file = st.file_uploader(
            "Upload Resume (PDF or TXT)", type=["pdf", "txt"]
        )
        previous_resume = False
    else:
        resume_file = os.path.join("./parsed_files", resume_file)
        previous_resume = True
    resume_caching = st.checkbox("Caching", key="resume_caching", value=True)

    job_url = st.text_input(
        "Job URL", placeholder="https://www.linkedin.com/jobs/view/..."
    )
    job_text = st.text_area("Job Text", placeholder="Paste job description here")

    job_caching = st.checkbox("Caching", key="job_caching", value=True)
    analyze_button = st.button("Analyze Match")

    st.divider()
    # show crew .env config items
    st.markdown("### Crew .env config items:")
    st.markdown(f"""
    ```
    GEMINI_API_KEY: {os.getenv("GEMINI_API_KEY")}
    CREWAI_TRACING_ENABLED: {os.getenv("CREWAI_TRACING_ENABLED")}
    CREWAI_STORAGE_DIR: {os.getenv("CREWAI_STORAGE_DIR")}
    CREWAI_PLATFORM_INTEGRATION_TOKEN: {os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")}
    ```
    """)
    fake_result = st.selectbox("FAKE_RESULT", ["True", "False"], index=1)
    if fake_result == "True":
        os.environ["FAKE_RESULT"] = "True"
    else:
        os.environ["FAKE_RESULT"] = "False"

if analyze_button:
    if resume_file is not None and (job_url or job_text):
        if job_text:
            # write to a temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", dir="jobs"
            ) as tmp_file:
                tmp_file.write(job_text.encode())
                tmp_file_path = tmp_file.name
            job_url = tmp_file_path
        with st.spinner("Preparing files..."):
            # Save uploaded file to a temporary location
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
                resume_analysis, resume_analysis_path = (
                    resume_job_scorer.resume_skill_analyser_crew(
                        tmp_file_path, resume_caching
                    )
                )
                st.success(
                    f"**Resume analysis complete!**\n\n"
                    f"Analysis saved to: `{resume_analysis_path}`\n\n"
                    f"Resume stored as: `./parsed_files/{resume_parsed_filename}.txt`"
                )
            with st.spinner("Analysing job..."):
                job_analysis, job_analysis_path = (
                    resume_job_scorer.job_skill_analyser_crew(job_url, job_caching)
                )
                st.success(
                    f"**Job analysis complete!**\n\nAnalysis saved to: `{job_analysis_path}`"
                )
            with st.spinner("Deciding..."):
                final_decision, final_decision_path = (
                    resume_job_scorer.compare_and_decide_crew(
                        resume_analysis.raw, job_analysis.raw, job_caching
                    )
                )
                st.success(
                    f"**Final decision complete!**\n\nDecision analysis saved to: `{final_decision_path}`"
                )

            # Parse the output
            try:
                result_json = utils.extract_json_from_crew_output(final_decision.raw)
            except Exception as e:
                st.error(f"Failed to parse result JSON: {e}")
                st.text(final_decision.raw)
                st.stop()

            # Display Results
            st.header("Analysis Results")

            # Job Details
            if "job_posting_details" in result_json:
                details = result_json["job_posting_details"]
                org = details.get("organization")
                url = details.get("job_url")
                source = details.get("job_source")
                role_summary = details.get("role_summary")

                st.subheader("Job Details")
                st.write(
                    f"""
                    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; margin-bottom: 15px;">
                        <h3 style="margin-top: 0; color: #333;">{org}</h3>
                        <p style="margin-bottom: 10px;"><strong>Source:</strong> {source}</p>
                        <p style="margin-bottom: 10px;"><strong>Role Summary:</strong> {role_summary}</p>
                        <a href="{url}" target="_blank" style="text-decoration: none;">
                            <div style="display: inline-block; background-color: #007bff; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;">
                                View Job Posting
                            </div>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if "resume_vs_job_decision" in result_json:
                decision = result_json["resume_vs_job_decision"]
                st.subheader("Resume vs Job Decision")
                if decision["decision"] == "Pass":
                    st.success(
                        f"Candidate resume is a good fit for the Job! Reason: {decision['reason']}"
                    )
                else:
                    st.error(
                        f"Candidate resume is not a good fit for the Job! Reason: {decision['reason']}"
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

            with col4:
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

            st.divider()

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    else:
        st.warning("Please upload a resume and provide a job URL.")
