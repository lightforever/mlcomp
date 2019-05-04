import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';

import {Observable, of} from 'rxjs';
import {catchError, map, tap} from 'rxjs/operators';

import {Dag, PaginatorRes} from './models';
import {MessageService} from './message.service';
import {AppSettings} from './app-settings'

const httpOptions = {
  headers: new HttpHeaders({'Content-Type': 'application/json'})
};

@Injectable({providedIn: 'root'})
export class DagService {

  private url = `${AppSettings.API_ENDPOINT}dags`;  // URL to web api

  constructor(private http: HttpClient,
              private messageService: MessageService) {
  }

  /** GET dags from the server */
  getDags(sortColumn: string, sortDescending: boolean, pageNumber: number, pageSize: number, filter: string, project: number): Observable<PaginatorRes<Dag>> {
    return this.http.get<PaginatorRes<Dag>>(`${this.url}?sort_column=${sortColumn}&sort_descending=${sortDescending}&page_number=${pageNumber}&page_size=${pageSize}&filter=${filter}&project=${project}`)
      .pipe(
        tap(_ => this.log('fetched dags')),
        catchError(this.handleError<PaginatorRes<Dag>>('getDags', new PaginatorRes<Dag>()))
      );
  }
  
  /** GET dag by id. Will 404 if id not found */
  getDag(id: number): Observable<Dag> {
    const url = `${this.url}/${id}`;
    return this.http.get<Dag>(url).pipe(
      tap(_ => this.log(`fetched dag id=${id}`)),
      catchError(this.handleError<Dag>(`getDag id=${id}`))
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
      console.error(error); // log to console instead

      this.log(`${operation} failed: ${error.message}`);

      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }

  /** Log a DagService message with the MessageService */
  private log(message: string) {
    this.messageService.add(`DagService: ${message}`);
  }
}
