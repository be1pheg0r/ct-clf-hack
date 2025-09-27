export interface ResultData {
  path_to_study?: string;
  study_uid?: string;
  series_uid?: string;
  probability_of_pathology?: number | string; // бекенд может вернуть строку => приводим к number при использовании
  pathology?: number | string; // 0 or 1
  processing_status?: string;
  time_of_processing?: string | number;
}
