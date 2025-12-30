import os
from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
from pathlib import Path
from lib import utils
from pydantic import BaseModel
from crewai.llm import LLM
from crewai.crews.crew_output import CrewOutput
import datetime
import json

openai_llm = LLM(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,  # Lower temperature for more consistent results.
)

empty_crew_usage_metrics = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "cached_prompt_tokens": 0,
    "completion_tokens": 0,
    "successful_requests": 0,
}


# pydantic models for output
class ResumeSkills(BaseModel):
    resume_skills: list[str]
    years_of_experience: int
    certifications: list[str]
    security_clearances: list[str]


class JobRequirements(BaseModel):
    organization: str
    job_summary: str
    years_of_experience: int
    required_skills: list[str]
    preferred_skills: list[str]
    required_certifications: list[str]
    required_security_clearances: list[str]


class JobScore(BaseModel):
    final_score: float
    required_skill_match_score: float
    preferred_skill_match_score: float
    matching_required_skills_count: int
    missing_required_skills_count: int
    matching_preferred_skills_count: int
    missing_preferred_skills_count: int
    total_required_skills_count: int
    total_preferred_skills_count: int
    matching_certifications_count: int
    missing_certifications_count: int
    matching_security_clearances_count: int
    missing_security_clearances_count: int


class JobVsResume(BaseModel):
    matching_required_skills: list[str]
    missing_required_skills: list[str]
    matching_preferred_skills: list[str]
    missing_preferred_skills: list[str]
    matching_certifications: list[str]
    missing_certifications: list[str]
    matching_security_clearances: list[str]
    missing_security_clearances: list[str]
    score: JobScore
    organization: str
    years_of_experience: int
    job_summary: str
    decision: str
    reason: str


# Tools for agents to use
class TextExtractor(BaseTool):
    def _run(self, text_source: str) -> str:
        try:
            if Path(text_source).exists():
                if text_source.endswith(".pdf"):
                    print(f"Extracting text from PDF: {text_source}")
                    return utils.extract_text_from_pdf(text_source)
                elif text_source.endswith(".txt"):
                    print(f"Extracting text from TXT: {text_source}")
                    return utils.extract_text_from_file(text_source)
            elif text_source.startswith("http"):
                print(f"Extracting text from URL: {text_source}")
                return utils.get_text_from_url(text_source)
            else:
                return text_source
        except Exception:
            # exception is thrown if it is already text and Path will throw an exception
            print("Assuming the text source is already raw text")
            return text_source


@CrewBase
class ResumeCrew:
    agents_config = "config/resume_agents.yaml"
    tasks_config = "config/resume_tasks.yaml"

    @agent
    def resume_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["resume_agent"],
        )

    @task
    def resume_skill_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["resume_skill_analysis"],
            output_pydantic=ResumeSkills,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            llm=openai_llm,
        )


# CrewBase classes to pull from yaml config files
@CrewBase
class JobCrew:
    agents_config = "config/job_agents.yaml"
    tasks_config = "config/job_tasks.yaml"

    @agent
    def job_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["job_agent"],
        )

    @task
    def job_skill_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["job_requirements_analysis"],
            output_pydantic=JobRequirements,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            llm=openai_llm,
        )


@CrewBase
class HRCrew:
    agents_config = "config/hr_agents.yaml"
    tasks_config = "config/hr_tasks.yaml"

    @agent
    def hr_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hr_agent"],
        )

    @task
    def resume_to_job_match_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["resume_to_job_match_analysis"],
            # output_pydantic=JobVsResume,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            llm=openai_llm,
        )


# Helper functions
def result_to_json(result: CrewOutput) -> str:
    if result.pydantic is None:
        result_str = result.raw
        try:
            result_json = json.loads(result_str)
            return result_json
        except Exception:
            return result_str
    else:
        result_pyd = result.pydantic
        result_json = result_pyd.model_dump()
        return result_json


# save crew output to file
def save_resume_skill_analysis(
    crew_object: Crew, crew_result: CrewOutput, candidate_info: dict
):
    """
    Save the resume skill analysis crew output to a file.

    Args:
        crew_object: The crew object.
        crew_result: The crew result.
        candidate_info: The candidate information.
    """
    print("Writing Resume skill analysis to files")
    candidate_name = candidate_info.get("name", "Unknown").replace(" ", "_")
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resume_analysis_filename = f"resume_skills_analysis_{candidate_name}_{datestr}.txt"
    save_result(crew_object, crew_result, resume_analysis_filename)
    return resume_analysis_filename


def get_job_file_name(crew_result: CrewOutput, job_details: dict):
    job_skill_analysis_result_json = result_to_json(crew_result)
    job_org = job_skill_analysis_result_json.get("organization", "Unknown")
    job_id = job_details.get("job_id", "Unknown")
    job_source = job_details.get("job_source", "Unknown")
    job_org = job_org.replace(" ", "_")
    job_id = job_id.replace(" ", "_")
    job_source = job_source.replace(" ", "_")
    job_filename = f"{job_source}_{job_org}_{job_id}.txt"
    return job_filename


def save_job_requirements_analysis(
    crew_object: Crew, crew_result: CrewOutput, job_details: dict, prefix: str
):
    job_filename = prefix + "_" + get_job_file_name(crew_result, job_details)
    print(f"Writing {prefix} task output")
    save_result(crew_object, crew_result, job_filename)
    return job_filename


def save_job_text(job_text: str, crew_result: CrewOutput):
    job_filename = get_job_file_name(crew_result)
    job_storage_dir = os.getenv("JOB_STORAGE_DIR")
    with open(f"{job_storage_dir}/{job_filename}", "w") as f:
        f.write(job_text)
    return job_filename


# save
def save_result(
    crew_object: Crew, crew_result: CrewOutput, file_name: str, job_details: dict = None
):
    """
    Generic save function to save the crew output to a file.

    Args:
        crew_object: The crew object.
        crew_result: The crew result.
        file_name: The file name.
        job_details: The job details.
    """
    crew_output_storage_dir = os.getenv("CREW_OUTPUT_STORAGE_DIR")
    with open(f"{crew_output_storage_dir}/{file_name}", "w") as f:
        f.write("Crew Statistics\n")
        f.write(json.dumps(crew_object.usage_metrics.__dict__, indent=4))
        f.write("\n")
        f.write("---------\n")
        for i, task_output in enumerate(crew_result.tasks_output):
            f.write(f"-------------- {i + 1}. {task_output.agent} --------------\n")
            f.write(f"Description:\n{task_output.description}\n")
            f.write(f"Summary:\n{task_output.summary}\n")
            for mesg in task_output.messages:
                f.write(f"{mesg['role']}: {mesg['content']}\n")
            f.write(f"Raw:\n{task_output.raw}\n")
            f.write("\n")
        if job_details:
            f.write("Job Details\n")
            f.write(json.dumps(job_details, indent=4))
            f.write("\n")
    return file_name


# Crew execution wrapper functions
def resume_skill_analyzer_crew(
    resume_source: str, get_from_cache: bool = True
) -> tuple[CrewOutput, str, dict]:
    """
    Analyze the resume skills using the crew.

    Args:
        resume_source: The resume source.
        get_from_cache: Whether to get from cache.

    Returns:
        tuple[ResumeSkills, str, dict]: The resume skills, file name, and crew usage metrics.
    """
    resume_text = utils.extract_text_from_various_sources(resume_source)
    # extract name, email, phone number from resume text
    candidate_info = utils.nlp_parse_resume_get_name_email_phone(resume_text)
    print(candidate_info)
    if get_from_cache and resume_text in utils.dc:
        print("Using cached resume result")
        cached_result = list(utils.dc[resume_text])
        crew_usage_metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }
        cached_result.append(crew_usage_metrics)
        return cached_result

    resume_crew = ResumeCrew().crew()
    start = utils.currenttimemillis()
    resume_result = resume_crew.kickoff(
        inputs={"resume_text": resume_source},
    )
    end = utils.currenttimemillis()
    print(f"Resume skill analysis took {end - start} ms")
    print("Caching result for resume")
    save_path = save_resume_skill_analysis(resume_crew, resume_result, candidate_info)
    crew_usage_metrics = resume_crew.usage_metrics.__dict__

    utils.dc[resume_text] = resume_result, save_path
    return resume_result, save_path, crew_usage_metrics


def job_requirements_analyzer_crew(
    job_source: str, get_from_cache: bool = True
) -> tuple[CrewOutput, str, dict]:
    """
    Analyze the job requirements using the crew.

    Args:
        job_source: The job source.
        get_from_cache: Whether to get from cache.

    Returns:
        tuple[JobRequirements, str, dict]:
             The JobRequirements (crew output),
             file name, and
             crew usage metrics.
    """
    job_text = utils.extract_text_from_various_sources(job_source)
    job_details = utils.identify_job_source(job_source)

    if get_from_cache and job_text in utils.dc:
        print("Using cached job requirements result")
        cached_result = list(utils.dc[job_text])
        cached_result.append(empty_crew_usage_metrics)
        return cached_result

    job_crew = JobCrew().crew()
    input_data = {"job_text": job_text}
    start = utils.currenttimemillis()
    job_result = job_crew.kickoff(
        inputs=input_data,
    )
    end = utils.currenttimemillis()
    print(f"Job requirements analysis took {end - start} ms")
    print("Caching result for job")
    save_path = save_job_requirements_analysis(
        job_crew, job_result, job_details, "job_requirements"
    )
    crew_usage_metrics = job_crew.usage_metrics.__dict__
    save_job_text(job_text, job_result, job_details)
    utils.dc[job_text] = job_result, save_path, job_details
    return job_result, save_path, job_details, crew_usage_metrics


def hr_analyzer_crew(
    job_description: str,
    resume: str,
    job_details: dict,
    us_citizen: bool,
    security_clearance: str,
    get_from_cache: bool = True,
) -> tuple[CrewOutput, str, dict]:
    """
    Analyze the job vs resume using the crew.

    Args:
        job_source: The job source.
        resume_source: The resume source.
        get_from_cache: Whether to get from cache.

    Returns:
        tuple[JobVsResume, str, dict]:
             The JobVsResume (crew output),
             file name, and
             crew usage metrics.
    """
    if (
        get_from_cache
        and (job_description, resume, us_citizen, security_clearance) in utils.dc
    ):
        print("Using cached hr result")
        cached_result = list(
            utils.dc[(job_description, resume, us_citizen, security_clearance)]
        )
        cached_result.append(empty_crew_usage_metrics)
        return cached_result
    input_data = {
        "job_description": job_description,
        "resume": resume,
        "us_citizen": us_citizen,
        "security_clearance": security_clearance,
        "job_url": job_details.get("job_url", "Unknown"),
        "job_id": job_details.get("job_id", "Unknown"),
        "job_source": job_details.get("job_source", "Unknown"),
    }
    hr_crew = HRCrew().crew()
    start = utils.currenttimemillis()
    hr_result = hr_crew.kickoff(
        inputs=input_data,
    )
    end = utils.currenttimemillis()
    print(f"Job vs resume analysis took {end - start} ms")
    print("Caching result for job vs resume")
    save_path = save_job_requirements_analysis(
        hr_crew, hr_result, job_details, "hr_analysis"
    )
    crew_usage_metrics = hr_crew.usage_metrics.__dict__

    utils.dc[(job_description, resume, us_citizen, security_clearance)] = (
        hr_result,
        save_path,
    )
    return hr_result, save_path, crew_usage_metrics
