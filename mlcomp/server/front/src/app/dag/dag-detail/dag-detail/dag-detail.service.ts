import {Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {catchError, map, tap} from 'rxjs/operators';
import {AppSettings} from '../../../app-settings'
import {CodeResult, Data, Graph} from "../../../models";
import {BaseService} from "../../../base.service";

@Injectable({providedIn: 'root'})
export class DagDetailService extends BaseService{
  protected collection_part: string;
  protected single_part: string;
  private url = `${AppSettings.API_ENDPOINT}`;

  get_config(dag_id: number): Observable<Data<string>> {
    return this.http.post<Data<string>>(`${this.url}config`, dag_id)
      .pipe(
        tap(_ => this.log('fetched config')),
        catchError(this.handleError<Data<string>>('config', new Data<string>()))
      );
  }

  get_code(dag_id: number): Observable<CodeResult> {
     return this.http.post<CodeResult>(`${this.url}code`, dag_id)
      .pipe(
        tap(_ => this.log('fetched code')),
        catchError(this.handleError<CodeResult>('config', {'items': []}))
      );
  }
  get_graph(dag_id: number): Observable<Graph> {
     return this.http.post<Graph>(`${this.url}graph`, dag_id)
      .pipe(
        tap(_ => this.log('fetched graph')),
        catchError(this.handleError<Graph>('graph', new Graph()))
      );
  }

}
