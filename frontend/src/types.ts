export interface ViewerData {
  frames: string[]; // data:image/png;base64,...
  loading?: boolean;
}

export interface ResultData {
  path_to_study?: string;
  study_uid?: string;
  series_uid?: string;
  probability_of_pathology?: number | string;
  pathology?: number | string;
  processing_status?: string;
  time_of_processing?: string | number;
  report_xlsx?: string; // data URI: data:...;base64,...
}
