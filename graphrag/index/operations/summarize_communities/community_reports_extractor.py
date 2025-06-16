# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import logging
import traceback
from dataclasses import dataclass
import json

from pydantic import BaseModel, Field

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT

log = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")
    community: int | float = Field(description="The community ID this report belongs to.")
    
    def model_post_init(self, __context):
        # Convert float community IDs to integers if they are whole numbers
        if isinstance(self.community, float) and self.community.is_integer():
            self.community = int(self.community)


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None
    community: int | None = None


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: ChatModel
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int
    _community_id: int | None

    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
        community_id: int | None = None,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500
        self._community_id = community_id

    async def __call__(self, input_text: str):
        """Call method definition.
        
        Args:
            input_text: The input text to generate a community report from
            
        Returns:
            CommunityReportsResult containing the generated report
        """
        # Convert community_id to int if it's a string or None
        community_id = int(self._community_id) if self._community_id is not None else 0
        log.debug(f"Using community_id: {community_id} (type: {type(community_id).__name__})")
        
        try:
            prompt = self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: input_text,
                MAX_LENGTH_KEY: str(self._max_report_length),
            })
            
            # Log the prompt being sent to the LLM
            log.debug(f"Sending prompt to LLM (first 500 chars):\n{prompt[:500]}...")
            
            try:
                log.info(f"Sending prompt to LLM (first 500 chars):\n{prompt[:500]}...")
                
                # First, get the raw response without JSON parsing
                try:
                    raw_response = await self._model.achat(
                        prompt,
                        json=False,  # Get raw response first
                        name="create_community_report",
                    )
                    
                    # Log the raw response
                    log.info(f"Raw response from LLM (first 2000 chars):\n{str(raw_response)[:2000]}...")
                    if hasattr(raw_response, 'text'):
                        log.info(f"Response text (first 2000 chars):\n{raw_response.text[:2000]}...")
                    
                    # Now try to parse as JSON
                    try:
                        # If the response has a text attribute, use that, otherwise use the string representation
                        if hasattr(raw_response, 'text'):
                            response_text = raw_response.text
                        elif hasattr(raw_response, 'content'):
                            response_text = raw_response.content
                        else:
                            response_text = str(raw_response)
                        
                        log.info(f"Response text (first 2000 chars):\n{response_text[:2000]}...")
                        
                        # Extract JSON from markdown code blocks if present
                        if '```' in response_text:
                            log.info("Found markdown code blocks in response, extracting JSON content...")
                            # Extract content between first ```json and next ```
                            if '```json' in response_text:
                                json_content = response_text.split('```json')[1].split('```')[0].strip()
                            else:
                                # If no ```json marker, just get content between first set of ```
                                json_content = response_text.split('```')[1].split('```')[0].strip()
                            log.info(f"Extracted JSON content: {json_content[:500]}...")
                        else:
                            json_content = response_text
                        
                        # Try to parse the response as JSON
                        import json
                        try:
                            # First try to parse as is
                            json_response = json.loads(json_content)
                            log.info("Successfully parsed response as JSON")
                        except json.JSONDecodeError as je:
                            log.error(f"Failed to parse response as JSON: {str(je)}")
                            log.error(f"Response content that failed to parse: {json_content[:2000]}")
                            # Try to clean the JSON content
                            try:
                                # Remove any non-JSON text before the first {
                                start_idx = json_content.find('{')
                                if start_idx >= 0:
                                    json_content = json_content[start_idx:]
                                    # Find the last }
                                    end_idx = json_content.rfind('}')
                                    if end_idx >= 0:
                                        json_content = json_content[:end_idx+1]
                                        json_response = json.loads(json_content)
                                        log.info("Successfully parsed cleaned JSON")
                                    else:
                                        raise ValueError("No closing brace found in JSON content")
                                else:
                                    raise ValueError("No opening brace found in JSON content")
                            except Exception as e:
                                log.error(f"Failed to clean and parse JSON: {str(e)}")
                                # Try to extract partial JSON if possible
                                try:
                                    # Look for any valid JSON object in the text
                                    import re
                                    json_objects = re.findall(r'\{[^{}]*\}', json_content)
                                    if json_objects:
                                        # Try each potential JSON object
                                        for obj in json_objects:
                                            try:
                                                json_response = json.loads(obj)
                                                if all(k in json_response for k in ['title', 'summary', 'rating', 'rating_explanation', 'findings']):
                                                    log.info("Successfully extracted partial JSON")
                                                    break
                                            except:
                                                continue
                                        else:
                                            raise ValueError("No valid JSON object found")
                                    else:
                                        raise ValueError("No JSON object pattern found")
                                except Exception as e2:
                                    log.error(f"Failed to extract partial JSON: {str(e2)}")
                                    raise ValueError(f"LLM response is not valid JSON: {str(je)}") from je
                        
                        # Ensure community ID is included in the response before validation
                        if isinstance(json_response, dict):
                            json_response['community'] = community_id
                            log.debug(f"Set community_id in JSON response: {json_response['community']}")
                            
                            # Ensure all required fields are present
                            required_fields = ['title', 'summary', 'rating', 'rating_explanation', 'findings']
                            missing_fields = [field for field in required_fields if field not in json_response]
                            if missing_fields:
                                log.warning(f"Missing required fields in JSON response: {missing_fields}")
                                # Add default values for missing fields
                                for field in missing_fields:
                                    if field == 'findings':
                                        json_response[field] = []
                                    elif field == 'rating':
                                        json_response[field] = 0.0
                                    else:
                                        json_response[field] = "No content available"
                        
                        # Convert to CommunityReportResponse model
                        output = CommunityReportResponse.model_validate(json_response)
                        log.info("Successfully created CommunityReportResponse")
                        return self._create_result(output)
                        
                    except Exception as e:
                        log.error(f"Error validating response against model: {str(e)}")
                        log.error(f"Response content that failed validation: {response_text[:2000]}")
                        # Try to create a partial response from available fields
                        try:
                            if isinstance(json_response, dict):
                                # Create a new response with available fields
                                partial_response = {
                                    'title': json_response.get('title', 'Generated Report'),
                                    'summary': json_response.get('summary', 'No content available'),
                                    'findings': json_response.get('findings', []),
                                    'rating': json_response.get('rating', 0.0),
                                    'rating_explanation': json_response.get('rating_explanation', 'No rating available'),
                                    'community': community_id
                                }
                                output = CommunityReportResponse.model_validate(partial_response)
                                log.info("Successfully created partial response from available fields")
                                return self._create_result(output)
                        except Exception as e2:
                            log.error(f"Error creating partial response: {str(e2)}")
                            raise
                        
                except Exception as e:
                    log.error(f"Error in LLM communication: {str(e)}", exc_info=True)
                    if hasattr(e, 'response'):
                        log.error(f"Error response object: {e.response}")
                        if hasattr(e.response, 'text'):
                            log.error(f"Error response text (first 2000 chars):\n{e.response.text[:2000]}")
                    raise
                
            except Exception as e:
                log.error(f"Error calling LLM: {str(e)}")
                log.error(f"Error type: {type(e).__name__}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    log.error(f"Error response text: {e.response.text}")
                raise
            
            # If we got here, it means the first parsing attempt failed
            log.warning("No valid response format found (missing both parsed_response and text)")
            # Initialize default output with community ID
            default_output_data = {
                'title': "Generated Report",
                'summary': "No content available",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "No rating available",
                'community': community_id
            }
            default_output = CommunityReportResponse.model_validate(default_output_data)
            return self._create_result(default_output)
            
        except Exception as e:
            log.exception(f"Error generating community report for community_id={community_id}")
            self._on_error(e, traceback.format_exc(), None)
            # Return a default result with error information and community ID
            error_output_data = {
                'title': "Error Generating Report",
                'summary': f"An error occurred: {str(e)}",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "Report generation failed",
                'community': community_id  # Use the processed community_id
            }
            error_output = CommunityReportResponse.model_validate(error_output_data)
            return self._create_result(error_output)

    def _create_result(self, output: CommunityReportResponse | None) -> CommunityReportsResult:
        """Helper method to create a consistent CommunityReportsResult.
        
        Args:
            output: The CommunityReportResponse to create a result from, or None
            
        Returns:
            CommunityReportsResult containing the output and metadata
        """
        # Handle None output
        if output is None:
            output = CommunityReportResponse.model_validate({
                'title': "Generated Report",
                'summary': "No content available",
                'findings': [],
                'rating': 0.0,
                'rating_explanation': "No rating available",
                'community': getattr(self, '_community_id', 0) or 0
            })
        
        # Ensure output has community ID set
        if hasattr(output, 'community') and output.community is None:
            output.community = getattr(self, '_community_id', 0) or 0
        
        # Generate text output
        text_output = self._get_text_output(output)
        
        # If text output is empty or error, try to create a basic report from available fields
        if text_output.startswith("Error:") or not text_output.strip():
            findings_text = "\n\n".join(
                f"## {f.summary}\n\n{f.explanation}"
                for f in (output.findings or [])
                if hasattr(f, 'summary') and hasattr(f, 'explanation')
            )
            text_output = f"# {output.title}\n\n{output.summary}\n\n{findings_text}"
        
        # Log the final output for debugging
        log.debug(f"Created result with title: {output.title}")
        log.debug(f"Summary length: {len(output.summary)}")
        log.debug(f"Number of findings: {len(output.findings)}")
        log.debug(f"Text output length: {len(text_output)}")
        
        # Create the result with the parsed output
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
            community=output.community
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        """Convert the report to a formatted text output."""
        if not report:
            return "Error: Invalid report format"
            
        try:
            # Format findings
            findings_text = "\n\n".join(
                f"## {f.summary}\n\n{f.explanation}" 
                for f in (report.findings or []) 
                if hasattr(f, 'summary') and hasattr(f, 'explanation')
            )
            
            # Format rating section
            rating_text = f"\n\nImpact Severity Rating: {report.rating}/10\n{report.rating_explanation}"
            
            # Combine all sections
            return f"# {getattr(report, 'title', 'Untitled Report')}\n\n{getattr(report, 'summary', '')}{rating_text}\n\n{findings_text}"
        except Exception as e:
            log.error(f"Error formatting report: {e}")
            # Try to create a basic report from available fields
            try:
                findings_text = "\n\n".join(
                    f"## {f.summary}\n\n{f.explanation}"
                    for f in (report.findings or [])
                    if hasattr(f, 'summary') and hasattr(f, 'explanation')
                )
                return f"# {getattr(report, 'title', 'Untitled Report')}\n\n{getattr(report, 'summary', '')}\n\n{findings_text}"
            except Exception as e2:
                log.error(f"Error creating basic report: {e2}")
                return "Error: Could not format report"
