import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';

import {Observable, of} from 'rxjs';
import {catchError, map, tap} from 'rxjs/operators';

import {Project} from './models';
import {MessageService} from './message.service';
import {AppSettings} from './app-settings'

const httpOptions = {
  headers: new HttpHeaders({'Content-Type': 'application/json'})
};

@Injectable({providedIn: 'root'})
export class ProjectService {

  private url = `${AppSettings.API_ENDPOINT}/projects`;  // URL to web api

  constructor(private http: HttpClient,
              private messageService: MessageService) {
  }

  /** GET projects from the server */
  getProjects(sortColumn: string, sortDescending: boolean, pageNumber: number, pageSize: number, filter: string): Observable<Project[]> {
    return this.http.get<Project[]>(`${this.url}?sort_column=${sortColumn}&sort_descending=${sortDescending}&page_number=${pageNumber}&page_size=${pageSize}&filter=${filter}`)
      .pipe(
        tap(_ => this.log('fetched projects')),
        catchError(this.handleError<Project[]>('getProjects', []))
      );
  }

  /** GET project by id. Will 404 if id not found */
  getProject(id: number): Observable<Project> {
    const url = `${this.url}/${id}`;
    return this.http.get<Project>(url).pipe(
      tap(_ => this.log(`fetched project id=${id}`)),
      catchError(this.handleError<Project>(`getProject id=${id}`))
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
    this.messageService.add(`ProjectService: ${message}`);
  }
}
