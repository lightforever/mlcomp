import {Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {catchError, map, tap} from 'rxjs/operators';
import {AppSettings} from './app-settings'
import {CodeNode, Data, Graph} from "./models";
import {BaseService} from "./base.service";

@Injectable({providedIn: 'root'})
export class DagDetailService extends BaseService{
  protected collection_part: string;
  protected single_part: string;
  private url = `${AppSettings.API_ENDPOINT}`;

  get_config(dag_id: string): Observable<Data<string>> {
    return this.http.get<Data<string>>(`${this.url}config?dag=${dag_id}`)
      .pipe(
        tap(_ => this.log('fetched config')),
        catchError(this.handleError<Data<string>>('config', new Data<string>()))
      );
  }

  get_code(dag_id: string): Observable<CodeNode[]> {
     return this.http.get<CodeNode[]>(`${this.url}code?dag=${dag_id}`)
      .pipe(
        tap(_ => this.log('fetched code')),
        catchError(this.handleError<CodeNode[]>('config', []))
      );
  }
  get_graph(dag_id: string): Observable<Graph> {
     return this.http.get<Graph>(`${this.url}graph?dag=${dag_id}`)
      .pipe(
        tap(_ => this.log('fetched graph')),
        catchError(this.handleError<Graph>('graph', new Graph()))
      );
  }

}
