import streamlit as st
import os
import tempfile
import json
import resume_job_scorer
from lib import utils
from dotenv import load_dotenv

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
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_url = st.text_input(
        "Job URL", placeholder="https://www.linkedin.com/jobs/view/..."
    )
    analyze_button = st.button("Analyze Match")

if analyze_button:
    if resume_file is not None and job_url:
        with st.spinner("Analyzing match... This may take a minute."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resume_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Run the crew
                result = resume_job_scorer.run_flow(
                    resume=tmp_file_path, job_description=job_url
                )

                # Parse the output
                try:
                    result_json = utils.extract_json_from_crew_output(result.raw)
                except Exception as e:
                    st.error(f"Failed to parse result JSON: {e}")
                    st.text(result.raw)
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

                    result_log_file_name = resume_job_scorer.save_result(result)
                    st.caption(f"Analysis log saved to: `{result_log_file_name}`")
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
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

    else:
        st.warning("Please upload a resume and provide a job URL.")
