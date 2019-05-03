import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';

import {Observable, of} from 'rxjs';
import {catchError, map, tap} from 'rxjs/operators';

import {MessageService} from './message.service';
import {AppSettings} from './app-settings'
import {Data} from "./models";

@Injectable({providedIn: 'root'})
export class DagDetailService {

  private url = `${AppSettings.API_ENDPOINT}`;  // URL to web api

  constructor(private http: HttpClient,
              private messageService: MessageService) {
  }

  /** GET projects from the server */
  get_config(dag_id: string): Observable<Data<string>> {
    return this.http.get<Data<string>>(`${this.url}/config?dag_id=${dag_id}`)
      .pipe(
        tap(_ => this.log('fetched config')),
        catchError(this.handleError<Data<string>>('config', new Data<string>()))
      );
  }

  /**
   * Handle Http operation that failed.
   * Let the app continue.
   * @param operation - name of the operation that failed
   * @param result - optional value to return as the observable result
   */
  private handleError<T>(operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {

      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead

      // TODO: better job of transforming error for user consumption
      this.log(`${operation} failed: ${error.message}`);

      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }

  /** Log a ProjectService message with the MessageService */
  private log(message: string) {
    this.messageService.add(`DagDetailService: ${message}`);
  }
}
