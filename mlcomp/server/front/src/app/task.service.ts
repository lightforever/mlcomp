import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

import { Observable, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';

import {PaginatorRes, Task} from './models';
import { MessageService } from './message.service';
import {AppSettings} from './app-settings'

@Injectable({ providedIn: 'root' })
export class TaskService {

  private url = `${AppSettings.API_ENDPOINT}`;

  constructor(
    private http: HttpClient,
    private messageService: MessageService) { }

  /** GET tasks from the server */
  get_tasks(dag_id?:number): Observable<PaginatorRes<Task>> {
    return this.http.get<PaginatorRes<Task>>(`${this.url}/tasks?dag_id=${dag_id}`)
      .pipe(
        tap(_ => this.log('fetched tasks')),
        catchError(this.handleError<PaginatorRes<Task>>('get_tasks', new PaginatorRes<Task>()))
      );
  }

  /** GET task by id. Will 404 if id not found */
  get_task(id: number): Observable<Task> {
    const url = `${this.url}/task/${id}`;
    return this.http.get<Task>(url).pipe(
      tap(_ => this.log(`fetched task id=${id}`)),
      catchError(this.handleError<Task>(`get_task id=${id}`))
    );
  }

  /**
   * Handle Http operation that failed.
   * Let the app continue.
   * @param operation - name of the operation that failed
   * @param result - optional value to return as the observable result
   */
  private handleError<T> (operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {

      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead

      // TODO: better job of transforming error for user consumption
      this.log(`${operation} failed: ${error.message}`);

      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }

  /** Log a TaskService message with the MessageService */
  private log(message: string) {
    this.messageService.add(`TaskService: ${message}`);
  }
}
